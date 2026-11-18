"""Build notebooks/PathSnap_local.ipynb from the consolidated Kaggle notebook.

- Replaces the Kaggle setup cell with a local one (src/ already exists in the repo).
- Clears old outputs.
- Caps a few hard-coded sample sizes so the notebook also runs on a small
  stand-in dataset (the caps are no-ops on the real 80-trip train set).
Usage: python tools/make_local_notebook.py Sample_solution.ipynb notebooks/PathSnap_local.ipynb
"""
import json, sys

src, dst = sys.argv[1], sys.argv[2]
nb = json.load(open(src))

LOCAL_SETUP = '''# ===== LOCAL SETUP (replaces the Kaggle setup cell) =====
import os, sys
NB_DIR = os.path.abspath(".")
PROJECT = os.path.dirname(NB_DIR) if os.path.basename(NB_DIR) == "notebooks" else NB_DIR
assert os.environ.get("PATHSNAP_DATA"), "set PATHSNAP_DATA to the folder containing train/ and student_test/"
os.chdir(os.path.join(PROJECT, "notebooks"))
print("DATA_ROOT =", os.environ["PATHSNAP_DATA"]); print("cwd =", os.getcwd())
'''

PATCHES = [
    ("rng.choice(good_train, 15, replace=False)", "rng.choice(good_train, min(15, len(good_train)), replace=False)"),
    ("s = good.sample(200000, random_state=SEED)", "s = good.sample(min(200000, len(good)), random_state=SEED)"),
    ('.sample(200000, random_state=SEED).corr()', '.sample(min(200000, len(good)), random_state=SEED).corr()'),
    ("rng.choice(trip_ids, N_DECODER_TRIPS, replace=False)", "rng.choice(trip_ids, min(N_DECODER_TRIPS, len(trip_ids)), replace=False)"),
    ("gkf = GroupKFold(n_splits=4)", "gkf = GroupKFold(n_splits=min(4, train.trip_id.nunique()))"),
]

applied = {p[0]: 0 for p in PATCHES}
nb["cells"][1]["source"] = LOCAL_SETUP
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    c["outputs"], c["execution_count"] = [], None
    s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    for a, b in PATCHES:
        if a in s:
            s = s.replace(a, b); applied[a] += 1
    c["source"] = s
missing = [k for k, v in applied.items() if v == 0]
assert not missing, f"patches not applied: {missing}"
nb.setdefault("metadata", {})["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
json.dump(nb, open(dst, "w"), indent=1)
print("wrote", dst)
