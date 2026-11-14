"""Run ONCE by the project owner, in the virtual environment the models were trained in:
    python portable_demo/export_portable_models.py
Turns the saved scikit-learn models into plain JSON so the demo needs no packages.
The JSON is only written if it reproduces the scikit-learn predictions."""
import json, math, os, sys
import joblib
import numpy as np
import pandas as pd
import sklearn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
card = json.load(open(os.path.join(ROOT, "models", "model_card.json")))
print("scikit-learn in use:", sklearn.__version__)


def predict(m, row):
    z = m["intercept"] + sum(c * ((math.log1p(row[f]) - mu) / s)
                             for f, mu, s, c in zip(m["features"], m["mean"], m["scale"], m["coef"]))
    return 1 / (1 + math.exp(-z))


df = pd.read_csv(os.path.join(ROOT, "data", "processed", "all_repos.csv"))
os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
for src, out, feats in (("defect_lr_reduced.joblib", "reduced.json", card["features"]),
                        ("defect_lr_size_only.joblib", "size_only.json", card["size_features"])):
    pipe = joblib.load(os.path.join(ROOT, "models", src))
    scaler, lr = pipe[1], pipe[-1]
    m = {"features": feats, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
         "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
         "trained_on": card["trained_on"], "cutoff": card["cutoff"]}
    try:
        sample = df.sample(300, random_state=1)
        ref = pipe.predict_proba(sample[feats])[:, 1]
    except Exception as e:
        sys.exit("\nSTOP: could not check %s against scikit-learn (%s: %s).\n"
                 "The models were saved with a different scikit-learn version than the one running now (%s).\n"
                 "Run this script in the project's own virtual environment (.venv), where the versions match."
                 % (src, type(e).__name__, e, sklearn.__version__))
    mine = np.array([predict(m, r) for _, r in sample.iterrows()])
    diff = float(np.abs(ref - mine).max())
    if diff > 1e-9:
        sys.exit("\nSTOP: %s does not match scikit-learn (max difference %.2e). Nothing was written." % (out, diff))
    json.dump(m, open(os.path.join(HERE, "models", out), "w"), indent=2)
    print("%-16s written. Max difference from the scikit-learn model: %.2e" % (out, diff))