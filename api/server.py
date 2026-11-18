"""PathSnap FastAPI server (Laptop 1: model + API).

Run from the project root:
    uvicorn api.server:app --host 0.0.0.0 --port 8000

POST /predict  body = {"trip_id": "...", "track": [[lon,lat,t],...],
                       "nodes": [[lon,lat],...], "arcs": [[u,v],...]}
returns        {"trip_id", "route": [arc ids, compressed], "point_arcs": [arc per GPS point or -1], ...}
The reference route is never sent to or used by the server.
"""
import os, sys, time, logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap_pipeline import PathSnapModel  # noqa: E402
from pathsnap_core import compress  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")

ARTIFACTS = os.environ.get("PATHSNAP_ARTIFACTS", os.path.join(ROOT, "artifacts"))
model = PathSnapModel(ARTIFACTS)
log.info("loaded model from %s (features=%d, hmm=%s)", ARTIFACTS, len(model.features), model.hmm)

app = FastAPI(title="PathSnap map-matching API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TripRequest(BaseModel):
    trip_id: str
    track: list[list[float]]   # lon, lat, t
    nodes: list[list[float]]   # lon, lat
    arcs: list[list[int]]      # u, v


@app.get("/health")
def health():
    return {"status": "ok", "model": type(model.model).__name__}


@app.post("/predict")
def predict(req: TripRequest):
    t0 = time.time()
    log.info("request trip_id=%s gps_points=%d nodes=%d arcs=%d",
             req.trip_id, len(req.track), len(req.nodes), len(req.arcs))
    if not req.track or not req.nodes or not req.arcs:
        raise HTTPException(400, "track, nodes and arcs must be non-empty")
    trip = dict(trip_id=req.trip_id,
                track=pd.DataFrame(req.track, columns=["lon", "lat", "t"]),
                nodes=pd.DataFrame(req.nodes, columns=["lon", "lat"]),
                arcs=pd.DataFrame(req.arcs, columns=["u", "v"]).astype(np.int64),
                route=None)
    out = model.predict_trip(trip)
    pred = out["pred_hmm"]
    route = compress(pred)
    dt = time.time() - t0
    log.info("predicted trip_id=%s candidates=%d route_arcs=%d matched_points=%d/%d in %.2fs",
             req.trip_id, len(out["cand"]), len(route), int((pred >= 0).sum()), len(pred), dt)
    return {"trip_id": req.trip_id, "route": route, "point_arcs": [int(x) for x in pred],
            "n_points": len(pred), "seconds": round(dt, 3)}
