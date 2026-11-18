"""Folium helpers (report requirement: GPS + road network + reference route + predicted route)."""
import numpy as np
import folium
from pathsnap_core import compress


def arc_coords(geo, arc_ids):
    """[[lat,lon],[lat,lon]] per arc (folium wants lat,lon)."""
    nl = geo.nodes_lonlat
    return [[[nl[geo.u[a], 1], nl[geo.u[a], 0]], [nl[geo.v[a], 1], nl[geo.v[a], 0]]] for a in arc_ids]


def route_polyline(geo, arcs):
    """Continuous polyline for a sequence of arcs."""
    nl = geo.nodes_lonlat
    pts = []
    for a in arcs:
        pts.append([nl[geo.u[a], 1], nl[geo.u[a], 0]])
        pts.append([nl[geo.v[a], 1], nl[geo.v[a], 0]])
    return pts


def make_trip_map(trip, geo, layers, out_html, gps_stride=3, net_pad_deg=0.002, max_net_arcs=6000):
    """layers: dict name -> dict(arcs=[...], color='red', weight=4, dash=None). Reference/predicted routes."""
    tr = trip["track"]
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=15, tiles="OpenStreetMap")
    # road network near the trajectory
    nl = geo.nodes_lonlat
    lon_lo, lon_hi = tr.lon.min() - net_pad_deg, tr.lon.max() + net_pad_deg
    lat_lo, lat_hi = tr.lat.min() - net_pad_deg, tr.lat.max() + net_pad_deg
    ulon, ulat = nl[geo.u, 0], nl[geo.u, 1]
    inb = np.where((ulon > lon_lo) & (ulon < lon_hi) & (ulat > lat_lo) & (ulat < lat_hi))[0]
    if len(inb) > max_net_arcs:
        inb = np.random.default_rng(0).choice(inb, max_net_arcs, replace=False)
    fg = folium.FeatureGroup(name="Road network (arcs)", show=False)
    folium.PolyLine(arc_coords(geo, inb), color="#888888", weight=1, opacity=0.7).add_to(fg)
    fg.add_to(m)
    # raw GPS
    fg = folium.FeatureGroup(name="Raw GPS points", show=True)
    for lat, lon in tr[["lat", "lon"]].to_numpy()[::gps_stride]:
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True, fill_opacity=0.7, weight=0).add_to(fg)
    fg.add_to(m)
    for name, L in layers.items():
        fg = folium.FeatureGroup(name=name, show=True)
        folium.PolyLine(route_polyline(geo, L["arcs"]), color=L.get("color", "red"), weight=L.get("weight", 4),
                        opacity=0.85, dash_array=L.get("dash"), tooltip=name).add_to(fg)
        fg.add_to(m)
    m.fit_bounds([[lat_lo + net_pad_deg, lon_lo + net_pad_deg], [lat_hi - net_pad_deg, lon_hi - net_pad_deg]])
    legend = """<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
        padding: 8px 12px; border: 1px solid #666; font-size: 13px;"><b>Legend</b><br>
        <span style='color:blue'>&#9679;</span> Raw GPS<br>""" + "".join(
        f"<span style='color:{L.get('color','red')}'>&#9632;</span> {n}<br>" for n, L in layers.items()) + \
        "<span style='color:#888'>&#9632;</span> Road network</div>"
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    return m
