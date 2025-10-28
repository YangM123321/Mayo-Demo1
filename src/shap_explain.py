# src/shap_explain.py
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, shap
import matplotlib.pyplot as plt

OUT = Path("out")
MODELS = Path("models")
PLOTS = OUT / "shap"
PLOTS.mkdir(parents=True, exist_ok=True)

# ---- Load features like training ----
df = pd.read_parquet(OUT / "labs_curated.parquet")

# Pivot to wide table by LOINC per encounter
feat = (
    df.pivot_table(
        index=["patient_id", "encounter_id"],
        columns="loinc",
        values="lab_value",
        aggfunc="mean",
    )
    .reset_index()
    .rename_axis(None, axis=1)
    .fillna(0.0)
)

# Keep the same feature order you trained with
if {"2345-7", "718-7"}.issubset(feat.columns):
    X_df = feat[["2345-7", "718-7"]]
else:
    X_df = feat.iloc[:, 2:]  # all numeric after the two id columns

X = X_df.values
feature_names = list(X_df.columns)

# ---- Load model (RandomForest recommended for TreeExplainer) ----
# Falls back to LR if RF is not present
model_path_rf = MODELS / "admit_rf.joblib"
model_path_lr = MODELS / "admit_lr.joblib"
if model_path_rf.exists():
    model = joblib.load(model_path_rf)
    model_name = "random_forest"
else:
    model = joblib.load(model_path_lr)
    model_name = "logreg"

# ---- Compute SHAP values with a robust shape-normalization ----
# Prefer TreeExplainer for tree models, KernelExplainer for linear models as fallback
try:
    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(X)
    ev_raw = explainer.expected_value
except Exception:
    # Fallback for non-tree models
    background = shap.sample(X, min(200, len(X))) if len(X) > 200 else X
    explainer = shap.KernelExplainer(lambda data: model.predict_proba(data)[:, 1], background)
    sv_raw = explainer.shap_values(X, nsamples="auto")
    ev_raw = explainer.expected_value

def to_scalar_base(ev):
    """Return a scalar base value from any SHAP expected_value structure."""
    if isinstance(ev, (list, tuple, np.ndarray)):
        ev = np.array(ev)
        ev = np.ravel(ev)[-1]
    return float(ev)

def pick_2d_shap(sv, n_samples, n_features):
    """
    Normalize shap values to shape (n_samples, n_features).
    Handles:
      - list of arrays (take last output)
      - ndarray (n_samples, n_features)
      - ndarray (n_outputs, n_samples, n_features)
      - ndarray (n_samples, n_outputs, n_features)
    """
    if isinstance(sv, list):
        arr = np.asarray(sv[-1])
    else:
        arr = np.asarray(sv)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape == (2, n_samples, n_features) or arr.shape[0] in (1, 2):
            return arr[-1, :, :]
        if arr.shape[0] == n_samples and arr.shape[2] == n_features:
            return arr[:, -1, :]
        arr2 = np.squeeze(arr)
        if arr2.ndim == 2 and arr2.shape == (n_samples, n_features):
            return arr2
        raise ValueError(f"Unexpected SHAP shape {arr.shape} for (samples, features)")
    arr2 = np.squeeze(arr)
    if arr2.ndim == 2:
        return arr2
    raise ValueError(f"Unexpected SHAP ndim={arr.ndim}")

n_samples, n_features = X.shape
sv_all = pick_2d_shap(sv_raw, n_samples, n_features)
base = to_scalar_base(ev_raw)

# ---- Build Explanation objects ----
exp_all = shap.Explanation(
    values=sv_all,
    base_values=np.full(n_samples, base),
    data=X,
    feature_names=feature_names,
)

# 1) Global importance bar plot
plt.figure()
shap.plots.bar(exp_all, show=False, max_display=min(20, n_features))
plt.tight_layout()
plt.savefig(PLOTS / "shap_global_bar.png", dpi=150)
plt.close()

# 2) Per-sample explanation
i = 0  # choose first row; change as needed
sv0 = sv_all[i].astype(float)
x0 = X[i]
exp_i = shap.Explanation(
    values=sv0,
    base_values=base,
    data=x0,
    feature_names=feature_names,
)

# Try a modern waterfall; fallback to barh if needed
saved_waterfall = True
try:
    plt.figure()
    shap.plots.waterfall(exp_i, show=False, max_display=min(20, n_features))
    plt.tight_layout()
    plt.savefig(PLOTS / "shap_waterfall_first.png", dpi=150)
    plt.close()
except Exception as e:
    saved_waterfall = False
    plt.figure()
    order = np.argsort(np.abs(sv0))[::-1]
    plt.barh([feature_names[j] for j in order], sv0[order])
    plt.gca().invert_yaxis()
    plt.title("SHAP (fallback) - first sample")
    plt.xlabel("SHAP value")
    plt.tight_layout()
    plt.savefig(PLOTS / "shap_waterfall_first.png", dpi=150)
    plt.close()
    (PLOTS / "shap_waterfall_first.fallback.txt").write_text(str(e))

# 3) Optional force plot HTML (best effort)
saved_force = True
try:
    force_obj = shap.plots.force(exp_i, show=False)  # requires JS in a browser
    shap.save_html(str(PLOTS / "shap_force_first.html"), force_obj)
except Exception as e:
    saved_force = False
    (PLOTS / "shap_force_first.ERROR.txt").write_text(str(e))

print("Wrote:")
print(" ", PLOTS / "shap_global_bar.png")
print(" ", PLOTS / "shap_waterfall_first.png", "(fallback used)" if not saved_waterfall else "")
print(" ", PLOTS / "shap_force_first.html" if saved_force else "  (force plot skipped; see ERROR.txt)")
