"""Stand-in evaluator (Laptop 2). Sends trips to the API and scores the returned route.

    python api/evaluator_client.py --api http://<laptop1-ip>:8000 --data data/evaluator_test [trip_id ...]

Reference .route files stay on this side; they are only used for scoring.
"""
import argparse, os, sys, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402
from pathsnap_core import load_trip, list_trips, TripGeometry  # noqa: E402
from pathsnap_pipeline import score_prediction  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://127.0.0.1:8000")
ap.add_argument("--data", default=os.path.join(ROOT, "data", "evaluator_test"))
ap.add_argument("trips", nargs="*")
a = ap.parse_args()

print("health:", requests.get(a.api + "/health", timeout=10).json())
rows = []
for tid in a.trips or list_trips(a.data):
    trip = load_trip(a.data, tid)
    body = dict(trip_id=tid, track=trip["track"].values.tolist(), nodes=trip["nodes"].values.tolist(),
                arcs=trip["arcs"].values.tolist())
    t0 = time.time()
    r = requests.post(a.api + "/predict", json=body, timeout=600)
    r.raise_for_status()
    res = r.json()
    geo = TripGeometry(trip)
    m, _ = score_prediction(geo, np.array(res["point_arcs"]), trip["route"])
    rows.append((tid, geo.n_gps, m["pointwise_acc_all"], m["arc_f1"], m["route_continuity"], time.time() - t0))
    print(f"{tid}  pts={geo.n_gps:6d}  acc={m['pointwise_acc_all']:.4f}  arcF1={m['arc_f1']:.4f}  "
          f"continuity={m['route_continuity']:.4f}  ({rows[-1][-1]:.1f}s, HTTP {r.status_code})")
print(f"\nMEAN over {len(rows)} trips: acc={np.mean([r[2] for r in rows]):.4f}  arcF1={np.mean([r[3] for r in rows]):.4f}")
