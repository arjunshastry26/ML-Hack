"""
Central configuration.  >>> EDIT DATA_ROOT ONCE <<<  (or set env var PATHSNAP_DATA)

DATA_ROOT must point at the UNZIPPED 'Student folder' that contains:
    train/           (80 trips)
    student_test/    (10 trips)
Evaluator data is NEVER placed here - it is only given to the API at final evaluation.
"""
import os

DATA_ROOT = os.environ.get("PATHSNAP_DATA", r"C:\PathSnap\Student folder")   # <-- EDIT ME

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "student_test")

ARTIFACTS = os.path.join(PROJECT_ROOT, "artifacts")
DATA_OUT = os.path.join(ARTIFACTS, "data")
OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")
FIG_DIR = os.path.join(OUTPUTS, "figures")
MAP_DIR = os.path.join(OUTPUTS, "maps")
for _d in (ARTIFACTS, DATA_OUT, OUTPUTS, FIG_DIR, MAP_DIR):
    os.makedirs(_d, exist_ok=True)

TRAIN_TABLE = os.path.join(DATA_OUT, "train_candidates.parquet")
TEST_TABLE = os.path.join(DATA_OUT, "student_test_candidates.parquet")
QUALITY_CSV = os.path.join(ARTIFACTS, "trip_quality.csv")

SEED = 42

# ---- candidate generation / labelling (fixed in notebook 02 from TRAIN data only) ----
RADIUS_M = 50.0          # candidate search radius
K_MAX = 15               # max candidates per GPS point
SKIP_PEN = 1.0           # DP alignment: metres charged per skipped reference arc
QUALITY_MAX_MEDIAN_RESID_M = 30.0   # trips whose reference route does not fit their GPS are not used for training

# ---- model experimentation ----
STRIDE = 6               # use every 6th GPS point of the training trips for model comparison (1 Hz data is very redundant)
N_SPLITS = 3             # GroupKFold folds (by trip) for comparison
STRIDE_FINAL = 2         # denser sampling for the final refit

BASELINE_LGBM = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=SEED,
                     verbose=-1, n_jobs=-1)
