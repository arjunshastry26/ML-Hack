# PathSnap: TA dry run of the sample solution

What's here:
- `Sample_solution.ipynb`: the consolidated reference notebook, as provided (Kaggle version).
- `src/`: the project library, extracted from the notebook's first cell without changes.
- `notebooks/PathSnap_local.ipynb`: the same notebook adapted for a local run (`tools/make_local_notebook.py` builds it).
- `api/server.py`: FastAPI server for Laptop 1 (model + API). Not part of the material provided; written for the dry run.
- `api/evaluator_client.py`: a stand-in evaluator for Laptop 2. It sends trips, scores the returned routes and prints accuracy.

## 1. Install
```bash
pip install scikit-learn lightgbm optuna folium pyproj shapely pyarrow joblib fastapi uvicorn matplotlib scipy nbconvert ipykernel requests
```

## 2. Train (runs all 6 stages: EDA, features, 7-model comparison, Bayesian tuning, HMM/Viterbi, maps)
`PATHSNAP_DATA` must point at a folder that contains `train/` and `student_test/` (the student "Student folder").
```bash
export PATHSNAP_DATA="/path/to/Student folder"
cd notebooks && jupyter nbconvert --to notebook --execute PathSnap_local.ipynb --output PathSnap_local_executed.ipynb --ExecutePreprocessor.timeout=3600
```
Outputs go to `artifacts/` (`model_lgbm_tuned.joblib`, `pipeline_config.json`, `hmm_config.json`, `best_params.json`) and `outputs/` (CSVs, figures, Folium maps).

## 3. Laptop 1: start the API (from the project root)
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```
Each request logs one `request trip_id=...` line, one `predicted ...` line and an HTTP 200 line. These are the lines evaluators must check.

## 4. Laptop 2: send evaluator trips
```bash
python api/evaluator_client.py --api http://<laptop1-ip>:8000 --data data/evaluator_test
```
