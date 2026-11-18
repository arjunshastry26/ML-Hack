"""
PathSnap core library: I/O, projection, reference-label alignment, candidate
generation, feature engineering and HMM/Viterbi decoding.

Every notebook, the API and the evaluation scripts import from here so that
training-time and inference-time processing are IDENTICAL.
"""
import os
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import STRtree, linestrings, points as shp_points


# ----------------------------------------------------------------------------
# 1. I/O
# ----------------------------------------------------------------------------
def list_trips(partition_dir):
    """Return sorted trip ids (folder names) inside a partition folder."""
    return sorted(d for d in os.listdir(partition_dir)
                  if os.path.isdir(os.path.join(partition_dir, d)))


def load_trip(partition_dir, trip_id, with_route=True):
    """Load one trip. .route is optional so the same function works for inference."""
    base = os.path.join(partition_dir, trip_id, trip_id)
    track = pd.read_csv(base + ".track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(base + ".nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(base + ".arcs", sep=r"\s+", header=None, names=["u", "v"])
    route = None
    if with_route and os.path.exists(base + ".route"):
        route = pd.read_csv(base + ".route", header=None).iloc[:, 0].to_numpy(dtype=np.int64)
    return dict(trip_id=trip_id, track=track, nodes=nodes, arcs=arcs, route=route)


# ----------------------------------------------------------------------------
# 2. Local metric projection (azimuthal equidistant centred on the trip)
#    Each trip lives in a different city, so one global CRS is wrong; an AEQD
#    centred on the trip centroid keeps distances accurate to well under a
#    metre for trips spanning tens of km.
# ----------------------------------------------------------------------------
def make_projector(lon0, lat0):
    return Transformer.from_crs(
        4326, f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m +ellps=WGS84", always_xy=True)


class TripGeometry:
    """Projected view of one trip: GPS xy, arc endpoints, adjacency and a spatial index."""

    def __init__(self, trip):
        tr, nd, ar = trip["track"], trip["nodes"], trip["arcs"]
        self.trip_id = trip["trip_id"]
        self.proj = make_projector(tr.lon.mean(), tr.lat.mean())
        self.gx, self.gy = self.proj.transform(tr.lon.to_numpy(), tr.lat.to_numpy())
        self.gx, self.gy = np.asarray(self.gx), np.asarray(self.gy)
        self.t = tr.t.to_numpy(dtype=float)
        nx, ny = self.proj.transform(nd.lon.to_numpy(), nd.lat.to_numpy())
        nx, ny = np.asarray(nx), np.asarray(ny)
        self.u = ar.u.to_numpy(dtype=np.int64)
        self.v = ar.v.to_numpy(dtype=np.int64)
        self.ax, self.ay = nx[self.u], ny[self.u]          # arc start
        self.bx, self.by = nx[self.v], ny[self.v]          # arc end
        self.alen = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.n_nodes, self.n_arcs, self.n_gps = len(nd), len(ar), len(tr)
        self.nodes_lonlat = nd[["lon", "lat"]].to_numpy()
        self._tree = None
        # adjacency: arcs leaving each node (CSR style)
        order = np.argsort(self.u, kind="stable")
        self._out_sorted = order
        self._out_ptr = np.searchsorted(self.u[order], np.arange(self.n_nodes + 1))
        self.out_deg = np.diff(self._out_ptr)
        self.in_deg = np.bincount(self.v, minlength=self.n_nodes)
        # reverse-arc lookup: arc id of (v,u) or -1
        key = self.u * self.n_nodes + self.v
        rkey = self.v * self.n_nodes + self.u
        srt = np.argsort(key)
        pos = np.clip(np.searchsorted(key[srt], rkey), 0, len(key) - 1)
        hit = key[srt][pos] == rkey
        self.rev = np.where(hit, srt[pos], -1)

    def arcs_from_node(self, node):
        return self._out_sorted[self._out_ptr[node]:self._out_ptr[node + 1]]

    @property
    def tree(self):
        if self._tree is None:
            geoms = linestrings(np.stack([np.c_[self.ax, self.ay], np.c_[self.bx, self.by]], axis=1))
            self._tree = STRtree(geoms)
        return self._tree


def seg_dist(px, py, ax, ay, bx, by):
    """Distance from points to segments + projection fraction (vectorised, broadcastable)."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 1e-9, ((px - ax) * dx + (py - ay) * dy) / np.maximum(L2, 1e-9), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


# ----------------------------------------------------------------------------
# 3. Reference per-point labels.  .route gives an ARC sequence, NOT per-GPS-point
#    labels, so we align the GPS sequence to the reference arc sequence with a
#    monotone DP (stay/advance are free, skipping arcs costs `skip_pen` m per arc).
#    Used ONLY to create training labels / server-side scoring.
# ----------------------------------------------------------------------------
def align_points_to_route(geo, route, skip_pen=1.0, chunk=2000):
    m, n = len(route), geo.n_gps
    ax, ay, bx, by = geo.ax[route], geo.ay[route], geo.bx[route], geo.by[route]
    D = np.empty((n, m), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d, _ = seg_dist(geo.gx[s:e, None], geo.gy[s:e, None], ax[None], ay[None], bx[None], by[None])
        D[s:e] = d
    idx = np.arange(m, dtype=np.float32)
    ar = np.arange(m, dtype=np.int32)
    best = D[0] + skip_pen * idx                     # starting late is penalised
    bp = np.zeros((n, m), dtype=np.int32)
    for i in range(1, n):
        v = best - skip_pen * idx
        acc = np.minimum.accumulate(v)
        arg = np.maximum.accumulate(np.where(v == acc, ar, 0)).astype(np.int32)
        jump = np.full(m, np.inf, dtype=np.float32)
        jump[1:] = acc[:-1] + skip_pen * (idx[1:] - 1)   # k=j-1 -> zero penalty
        jarg = np.zeros(m, dtype=np.int32)
        jarg[1:] = arg[:-1]
        use_jump = jump < best
        bp[i] = np.where(use_jump, jarg, ar)
        best = np.where(use_jump, jump, best) + D[i]
    best = best + skip_pen * (m - 1 - idx)           # ending early is penalised
    j = int(np.argmin(best))
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = bp[i, j]
    resid = D[np.arange(n), path]
    return path, resid      # route[path[i]] = reference arc id of point i


# ----------------------------------------------------------------------------
# 4. Candidate generation + features
# ----------------------------------------------------------------------------
FEATURES = [
    "dist", "dist_minus_min", "rank", "n_cands", "proj_frac", "arc_len",
    "speed", "dt_prev", "hdg_cos", "hdg_sin", "hdg_abs", "along_disp", "along_disp_norm",
    "dist_m10", "dist_m5", "dist_p5", "dist_p10", "dist_win_mean", "dist_win_max",
    "has_rev", "rev_in_cands", "out_deg_v", "in_deg_u", "dist_to_end", "dist_to_start",
]


def generate_candidates(geo, radius=50.0, k_max=15):
    """For every GPS point return arcs within `radius` m (nearest k_max kept)."""
    pts = shp_points(np.c_[geo.gx, geo.gy])
    pi, ai = geo.tree.query(pts, predicate="dwithin", distance=radius)
    d, fr = seg_dist(geo.gx[pi], geo.gy[pi], geo.ax[ai], geo.ay[ai], geo.bx[ai], geo.by[ai])
    df = pd.DataFrame(dict(pt=pi.astype(np.int64), arc=ai.astype(np.int64), dist=d, proj_frac=fr))
    df = df.sort_values(["pt", "dist"], kind="stable")
    df["rank"] = df.groupby("pt").cumcount()
    return df[df["rank"] < k_max].reset_index(drop=True)


def build_features(geo, cand):
    """Add engineered features to the candidate table (vectorised)."""
    n = geo.n_gps
    gx, gy, t = geo.gx, geo.gy, geo.t
    i = np.arange(n)

    def sh(o):
        return np.clip(i + o, 0, n - 1)

    dx = gx[sh(2)] - gx[sh(-2)]
    dy = gy[sh(2)] - gy[sh(-2)]
    dtm = np.maximum(t[sh(2)] - t[sh(-2)], 1e-3)
    step = np.hypot(dx, dy)
    speed = step / dtm
    dt_prev = np.r_[0.0, np.diff(t)]
    wx = gx[sh(5)] - gx[sh(-5)]
    wy = gy[sh(5)] - gy[sh(-5)]
    p, a = cand["pt"].to_numpy(), cand["arc"].to_numpy()
    ax, ay, bx, by = geo.ax[a], geo.ay[a], geo.bx[a], geo.by[a]
    L = np.maximum(geo.alen[a], 1e-6)
    ux, uy = (bx - ax) / L, (by - ay) / L
    mv = step[p]
    mvx = np.where(mv > 1e-6, dx[p] / np.maximum(mv, 1e-6), 0.0)
    mvy = np.where(mv > 1e-6, dy[p] / np.maximum(mv, 1e-6), 0.0)
    moving = speed[p] > 0.5
    out = cand.copy()
    out["dist_minus_min"] = out["dist"] - out.groupby("pt")["dist"].transform("min")
    out["n_cands"] = out.groupby("pt")["arc"].transform("size")
    out["arc_len"] = geo.alen[a]
    out["speed"] = speed[p]
    out["dt_prev"] = dt_prev[p]
    out["hdg_cos"] = np.where(moving, mvx * ux + mvy * uy, 0.0)
    sind = np.where(moving, mvx * uy - mvy * ux, 0.0)
    out["hdg_sin"] = sind
    out["hdg_abs"] = np.abs(sind)
    along = wx[p] * ux + wy[p] * uy
    wn = np.hypot(wx[p], wy[p])
    out["along_disp"] = along
    out["along_disp_norm"] = np.where(wn > 1.0, along / np.maximum(wn, 1e-6), 0.0)
    ds = []
    for o in (-10, -5, 5, 10):
        q = sh(o)[p]
        d, _ = seg_dist(gx[q], gy[q], ax, ay, bx, by)
        ds.append(d)
        out[f"dist_{'m' if o < 0 else 'p'}{abs(o)}"] = d
    allw = np.stack(ds + [out["dist"].to_numpy()], axis=1)
    out["dist_win_mean"] = allw.mean(1)
    out["dist_win_max"] = allw.max(1)
    out["has_rev"] = (geo.rev[a] >= 0).astype(np.int8)
    key = set(zip(p.tolist(), a.tolist()))
    out["rev_in_cands"] = [int(rr >= 0 and (pp, rr) in key)
                           for pp, rr in zip(p.tolist(), geo.rev[a].tolist())]
    out["out_deg_v"] = geo.out_deg[geo.v[a]]
    out["in_deg_u"] = geo.in_deg[geo.u[a]]
    out["dist_to_start"] = out["proj_frac"] * geo.alen[a]
    out["dist_to_end"] = (1 - out["proj_frac"]) * geo.alen[a]
    return out


def trip_candidate_table(trip, radius=50.0, k_max=15, with_labels=True, skip_pen=1.0):
    """Full per-trip pipeline -> (geometry, candidate table with features[, labels], ref arc/pt)."""
    geo = TripGeometry(trip)
    cand = build_features(geo, generate_candidates(geo, radius, k_max))
    cand["trip_id"] = trip["trip_id"]
    ref_arc = None
    if with_labels and trip.get("route") is not None:
        path, _ = align_points_to_route(geo, trip["route"], skip_pen)
        ref_arc = trip["route"][path]
        cand["label"] = (cand["arc"].to_numpy() == ref_arc[cand["pt"].to_numpy()]).astype(np.int8)
    return geo, cand, ref_arc


# ----------------------------------------------------------------------------
# 5. HMM / Viterbi
#    hidden state at time t = one of the candidate arcs of GPS point t
#    emission   = log classifier probability  (scaled by w_emit)
#    transition = f(network hops between arcs), U-turn penalty, gap reset
# ----------------------------------------------------------------------------
DEFAULT_HMM = dict(w_emit=1.0, hop_pen=1.0, unreach_pen=8.0, uturn_pen=6.0, stay_pen=0.0,
                   max_hops=4, gap_s=30.0, p_floor=1e-4)


class HopOracle:
    """Cached BFS: #hops from arc a to arc b (1 = b starts where a ends)."""

    def __init__(self, geo, max_hops):
        self.geo, self.H, self.cache = geo, max_hops, {}

    def reach(self, a):
        r = self.cache.get(a)
        if r is None:
            r = {}
            frontier = [int(a)]
            for h in range(1, self.H + 1):
                nxt = []
                for x in frontier:
                    for y in self.geo.arcs_from_node(self.geo.v[x]):
                        y = int(y)
                        if y not in r:
                            r[y] = h
                            nxt.append(y)
                frontier = nxt
            self.cache[a] = r
        return r


def viterbi_decode(geo, cand, prob, params=None):
    """cand: candidate table sorted by pt; prob: P(candidate correct) per row.
    Returns pred_arc[n_gps] (-1 where a point has no candidate) and diagnostics."""
    P = {**DEFAULT_HMM, **(params or {})}
    n = geo.n_gps
    pt = cand["pt"].to_numpy()
    arc = cand["arc"].to_numpy()
    lp = np.log(np.clip(prob, P["p_floor"], 1.0)) * P["w_emit"]
    starts = np.searchsorted(pt, np.arange(n + 1))
    oracle = HopOracle(geo, int(P["max_hops"]))
    pred = -np.ones(n, dtype=np.int64)
    steps = [i for i in range(n) if starts[i + 1] > starts[i]]
    if not steps:
        return pred, dict(resets=0, no_valid_transition=0)
    resets = dead = 0
    back = []
    prev_i = steps[0]
    delta = lp[starts[prev_i]:starts[prev_i + 1]].copy()
    for i in steps[1:]:
        pa = arc[starts[prev_i]:starts[prev_i + 1]]
        ca = arc[starts[i]:starts[i + 1]]
        gap = geo.t[i] - geo.t[prev_i]
        if gap > P["gap_s"]:                      # large gap: transition carries no information
            T = np.zeros((len(pa), len(ca)))
            resets += 1
        else:
            T = np.empty((len(pa), len(ca)))
            for r, a in enumerate(pa):
                reach = oracle.reach(a)
                ra = geo.rev[a]
                for c, b in enumerate(ca):
                    if a == b:
                        T[r, c] = -P["stay_pen"]
                    elif b == ra:
                        T[r, c] = -P["uturn_pen"]
                    elif b in reach:
                        T[r, c] = -P["hop_pen"] * reach[b]
                    else:
                        T[r, c] = -P["unreach_pen"]     # invalid (disconnected) -> heavy penalty, not -inf
            dead += int(np.all(T <= -P["unreach_pen"]))
        sc = delta[:, None] + T
        bp = sc.argmax(0)
        delta = sc[bp, np.arange(len(ca))] + lp[starts[i]:starts[i + 1]]
        back.append(bp)
        prev_i = i
    k = int(delta.argmax())
    seq = [k]
    for bp in reversed(back):
        k = int(bp[k])
        seq.append(k)
    seq = seq[::-1]
    for si, i in enumerate(steps):
        pred[i] = arc[starts[i] + seq[si]]
    return pred, dict(resets=resets, no_valid_transition=dead)


def classifier_only_decode(geo, cand, prob):
    """Argmax per point (observation-level top-1) -> pred_arc[n_gps]."""
    pred = -np.ones(geo.n_gps, dtype=np.int64)
    df = pd.DataFrame(dict(pt=cand["pt"].to_numpy(), arc=cand["arc"].to_numpy(), p=prob))
    top = df.loc[df.groupby("pt")["p"].idxmax()]
    pred[top["pt"].to_numpy()] = top["arc"].to_numpy()
    return pred


# ----------------------------------------------------------------------------
# 6. Route-level metrics
# ----------------------------------------------------------------------------
def compress(seq):
    out = []
    for x in seq:
        if x >= 0 and (not out or out[-1] != x):
            out.append(int(x))
    return out


def route_metrics(geo, pred, ref_arc, ref_route):
    """Pointwise accuracy, arc-set precision/recall/F1, continuity of predicted route."""
    valid = pred >= 0
    pointwise = float((pred[valid] == ref_arc[valid]).mean()) if valid.any() else 0.0
    pointwise_all = float((pred == ref_arc).mean())
    pr = compress(pred)
    sp, sr = set(pr), set(int(x) for x in ref_route)
    inter = len(sp & sr)
    prec = inter / max(len(sp), 1)
    rec = inter / max(len(sr), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    conn = float(np.mean([geo.v[a] == geo.u[b] for a, b in zip(pr[:-1], pr[1:])])) if len(pr) > 1 else 1.0
    return dict(pointwise_acc=pointwise, pointwise_acc_all=pointwise_all, arc_precision=prec,
                arc_recall=rec, arc_f1=f1, route_continuity=conn,
                n_pred_arcs=len(pr), n_ref_arcs=len(sr))


def top1_accuracy(cand, prob):
    """Observation-level top-1 accuracy over points with >=1 candidate."""
    df = pd.DataFrame(dict(pt=cand["pt"].to_numpy(), trip=cand["trip_id"].to_numpy(),
                           p=prob, y=cand["label"].to_numpy()))
    top = df.loc[df.groupby(["trip", "pt"])["p"].idxmax()]
    return float(top["y"].mean())
