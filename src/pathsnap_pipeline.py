"""
Inference pipeline shared by the notebooks (maps / evaluation) and the FastAPI backend.
predict_trip() NEVER sees reference routes: it takes a trip loaded with with_route=False.
score_prediction() is the only place the reference route is used (server side).
"""
import os, json, copy
import numpy as np
import pandas as pd
import joblib
from pathsnap_core import (TripGeometry, generate_candidates, build_features, viterbi_decode,
                           classifier_only_decode, align_points_to_route, route_metrics, compress,
                           DEFAULT_HMM)


class PathSnapModel:
    def __init__(self, artifacts_dir):
        self.model = joblib.load(os.path.join(artifacts_dir, "model_lgbm_tuned.joblib"))
        with open(os.path.join(artifacts_dir, "pipeline_config.json")) as f:
            self.cfg = json.load(f)
        with open(os.path.join(artifacts_dir, "hmm_config.json")) as f:
            self.hmm = {**DEFAULT_HMM, **json.load(f)}
        self.features = self.cfg["features"]

    def predict_trip(self, trip):
        """trip: dict from load_trip(..., with_route=False). Returns dict with geo, cand, prob, preds."""
        geo = TripGeometry(trip)
        cand = build_features(geo, generate_candidates(geo, self.cfg["radius_m"], self.cfg["k_max"]))
        cand["trip_id"] = trip["trip_id"]
        if len(cand) == 0:
            return dict(geo=geo, cand=cand, prob=np.array([]), pred_clf=-np.ones(geo.n_gps, int),
                        pred_hmm=-np.ones(geo.n_gps, int), info={})
        prob = self.model.predict_proba(cand[self.features])[:, 1]
        pred_clf = classifier_only_decode(geo, cand, prob)
        pred_hmm, info = viterbi_decode(geo, cand, prob, self.hmm)
        return dict(geo=geo, cand=cand, prob=prob, pred_clf=pred_clf, pred_hmm=pred_hmm, info=info)


def score_prediction(geo, pred, ref_route, skip_pen=1.0):
    """SERVER-SIDE ONLY: align GPS to the reference route and compare."""
    path, _ = align_points_to_route(geo, ref_route, skip_pen)
    ref_arc = ref_route[path]
    return route_metrics(geo, pred, ref_arc, ref_route), ref_arc


def truncate_geo(geo, n):
    g = copy.copy(geo)
    g.n_gps = n
    g.t = geo.t[:n]
    return g
