# PathSnap: TA dry run of the sample solution

What's here:
- `Sample_solution.ipynb`: the consolidated reference notebook, as provided (Kaggle version).
- `src/`: the project library, extracted from the notebook's first cell without changes.
- `notebooks/PathSnap_local.ipynb`: the same notebook adapted for a local run (`tools/make_local_notebook.py` builds it).
- `api/server.py`: FastAPI server for Laptop 1 (model + API). Not part of the material provided; written for the dry run.
- `api/evaluator_client.py`: a stand-in evaluator for Laptop 2. It sends trips, scores the returned routes and prints accuracy.

Data is **not** in the repo. You need `Student_folder.zip` and `evaluator_test.zip` from the faculty.

## Reference results (from a full run)
- Student_test: classifier only 91.1% → classifier + HMM **95.2%** pointwise accuracy
- 10 evaluator trips through the API: **95.35%** mean accuracy (trip 00000030 is the hardest at 81.5%)
- Full training takes about 10 minutes on 4 CPU cores

---

## Running it on your laptop

### 0. Prerequisites
Python 3.10 or newer (`python --version`) and git.

### 1. Get the code and the data
```
git clone https://github.com/arjunshastry26/ml-hack.git
cd ml-hack
git checkout claude/practical-pasteur-usupse
```
Unzip both zips into `data/` so the layout is:
```
ml-hack/data/Student folder/train/...           (80 trips)
ml-hack/data/Student folder/student_test/...    (10 trips)
ml-hack/data/evaluator_test/00000015/...        (10 trips)
```

### 2. Virtual environment + packages
macOS / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Windows (PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Train (about 10 minutes)
macOS / Linux:
```
export PATHSNAP_DATA="$(pwd)/data/Student folder"
cd notebooks
jupyter nbconvert --to notebook --execute PathSnap_local.ipynb --output PathSnap_local_executed.ipynb --ExecutePreprocessor.timeout=7200
cd ..
```
Windows (PowerShell):
```
$env:PATHSNAP_DATA = "$PWD\data\Student folder"
cd notebooks
jupyter nbconvert --to notebook --execute PathSnap_local.ipynb --output PathSnap_local_executed.ipynb --ExecutePreprocessor.timeout=7200
cd ..
```
To watch it cell by cell, run `jupyter notebook notebooks/PathSnap_local.ipynb` instead and use Run All. Set `PATHSNAP_DATA` in the same terminal first.

This creates `artifacts/` (the model and its config files) and `outputs/` (CSVs, figures, Folium maps you can open in a browser).

### 4. Start the API (Laptop 1), from the repo root
```
uvicorn api.server:app --host 0.0.0.0 --port 8000
```
Check it at http://127.0.0.1:8000/health. Interactive docs are at http://127.0.0.1:8000/docs.

### 5. Send evaluator trips (second terminal, or Laptop 2)
Same machine:
```
python api/evaluator_client.py --api http://127.0.0.1:8000 --data data/evaluator_test
```
Different laptop on the same Wi-Fi: on Laptop 1, find its IP (`ipconfig` on Windows, `ipconfig getifaddr en0` on macOS). Then on Laptop 2:
```
python api/evaluator_client.py --api http://<laptop1-ip>:8000 --data data/evaluator_test
```
You can pick specific trips, e.g. `... --data data/evaluator_test 00000030 00000094`.
If Laptop 2 can't connect, allow Python through Laptop 1's firewall. Campus Wi-Fi often blocks device-to-device traffic, so a phone hotspot is the usual fix.

In the API terminal, each request should show `request trip_id=...`, `predicted ... candidates=...` and `"POST /predict HTTP/1.1" 200 OK`. Those are the lines the guide says evaluators must check.
