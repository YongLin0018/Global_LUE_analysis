"""
Code for SHAP analysis
Author: Yong Lin
Email: linyong0018@igsnrr.ac.cn
Institution: Institute of Geographic Sciences and Natural Resources Research, CAS
Date: 2025-08-30

Description:
    Light-Use-Efficiency (LUE) modelling and interpretation pipeline.
    1. Trains Random-Forest regressors for three LUE targets.
    2. Evaluates models with 10-fold CV and an independent test split.
    3. Explains models with SHAP and produces publication-ready figures.
"""

# ───────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.patches import Patch

# ───────────────────────────────────────────────────────────────────────────────
# 1. Configuration  (change these paths to match your machine)
# ───────────────────────────────────────────────────────────────────────────────
FILE_PATH  = r"eexample data for ridge or SHAP analysis.xlsx"          # Example: raw data
OUTPUT_DIR = r"SHAP_Results"            # Example: output folder

# Feature grouping and colour scheme
GROUP_CONFIG = {
    'Climate':          ['FD', 'TA', 'SM', 'P', 'VPD'],
    'Canopy structure': ['FPAR', 'MLA', 'SLA'],
    'Canopy physiology':['CHL', 'GSL', 'LN', 'LP'],
    'Nutrient':         ['SOC', 'SN', 'SP', 'CNR']
}
COLOR_MAP = {
    'Climate':          '#74CEF7',
    'Canopy structure': '#31553B',
    'Canopy physiology':'#C7C51B',
    'Nutrient':         '#7072A6'
}
UNIFORM_COLOR = "#6A8E66"  # Neutral colour for scatter points / lines

# Three LUE targets to model
TARGETS = ['LUEmax', 'LUEinc', 'LUEact']

# Optional: unit labels for each predictor (used in plots if desired)
UNIT_DICT = {
    "GSL":"day","FD":"%","TA":"℃","SM":"%","P":"mm","VPD":"kPa",
    "SOC":"g/kg","SN":"g/kg","CNratio":"","FPAR":"","CHL":"μg/cm²",
    "MLA":"°","SLA":"m²/kg","LP":"mg/g","LN":"mg/g","SP":"mg/kg²"
}

# Global matplotlib settings
plt.rcParams.update({'font.size': 5})

# ───────────────────────────────────────────────────────────────────────────────
# 2. Helper utilities
# ───────────────────────────────────────────────────────────────────────────────
def get_color(feature: str) -> str:
    """
    Return the colour assigned to a feature based on its group.
    If the feature is not found in any group, a neutral grey is returned.
    """
    for grp, feats in GROUP_CONFIG.items():
        if feature in feats:
            return COLOR_MAP[grp]
    return '#999999'

# ───────────────────────────────────────────────────────────────────────────────
# 3. Model wrapper
# ───────────────────────────────────────────────────────────────────────────────
class RFWrapper(RandomForestRegressor):
    """
    Extends sklearn RandomForestRegressor to keep track of feature names
    when a pandas DataFrame is passed to `fit`.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_names_ = None

    def fit(self, X, y, **fit_params):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        return super().fit(X, y, **fit_params)

MODEL_NAME  = 'RF'
MODEL_CLASS = RFWrapper

# ───────────────────────────────────────────────────────────────────────────────
# 4. Plotting functions
# ───────────────────────────────────────────────────────────────────────────────
def plot_regression(ax, y_true, y_pred, cv_r2=None):
    """
    Scatter + regression line for observed vs predicted.
    cv_r2 is an optional dict with 'mean' and 'std' keys to annotate CV results.
    """
    r2 = r2_score(y_true, y_pred)
    slope, intercept, *_ = stats.linregress(y_true, y_pred)

    sns.regplot(
        x=y_true, y=y_pred,
        scatter_kws={'s': 15, 'color': UNIFORM_COLOR, 'alpha': 0.7},
        line_kws={'color': 'red', 'lw': 1}, ax=ax
    )
    # 1:1 line
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            'k--', lw=1)

    # Annotation
    txt = f"R² = {r2:.2f}\ny = {slope:.2f}x + {intercept:.2f}"
    if cv_r2:
        txt += f"\nCV R² = {cv_r2['mean']:.2f}±{cv_r2['std']:.2f}"
    ax.text(0.05, 0.85, txt, transform=ax.transAxes,
            ha='left', va='top', fontsize=5)
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')

def plot_shap_single(ax, feature: str, X_train, shap_vals, target):
    """
    Single-feature SHAP dependency plot.
    Includes LOWESS smoothing and Pearson correlation.
    """
    idx   = list(X_train.columns).index(feature)
    y_vals = shap_vals.values[:, idx]
    x_vals = X_train[feature].values

    # Background shading: red for negative SHAP, blue for positive
    ax.axhspan(0, np.max(y_vals), facecolor='#2C6BB3', alpha=0.15, zorder=0)
    ax.axhspan(np.min(y_vals), 0, facecolor='#A70505', alpha=0.15, zorder=0)

    # LOWESS smooth line
    smooth = lowess(y_vals, x_vals, frac=0.8, it=3)
    ax.scatter(x_vals, y_vals, s=5, color=UNIFORM_COLOR, alpha=0.7, zorder=2)
    ax.plot(smooth[:, 0], smooth[:, 1], 'k-', lw=0.6, zorder=2)

    ax.set_xlabel(feature)
    corr, p = stats.pearsonr(x_vals, y_vals)
    ax.annotate(f"r={corr:.2f}", xy=(0.05, 0.9),
                xycoords='axes fraction', ha='left', va='top', fontsize=5)

# ───────────────────────────────────────────────────────────────────────────────
# 5. Main pipeline
# ───────────────────────────────────────────────────────────────────────────────
def main():
    # 5.1 Load data -------------------------------------------------------------
    df = pd.read_excel(FILE_PATH).dropna()

    # Flatten feature list (only keep features present in the file)
    feats = [f for grp in GROUP_CONFIG.values() for f in grp if f in df.columns]

    # 5.2 Loop over each LUE target --------------------------------------------
    for target in TARGETS:
        X = df[feats]
        y = df[target]

        # 5.2.1 Split into 80 % train / 20 % independent test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42)

        # 5.2.2 10-fold cross-validation on training data
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = []
        for tr_idx, val_idx in cv.split(X_train):
            m = MODEL_CLASS(n_estimators=500, random_state=42, n_jobs=-1)
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            y_val = m.predict(X_train.iloc[val_idx])
            cv_scores.append(r2_score(y_train.iloc[val_idx], y_val))
        cv_r2 = {'mean': np.mean(cv_scores), 'std': np.std(cv_scores)}

        # 5.2.3 Train final model on full training set
        model = MODEL_CLASS(n_estimators=500, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # 5.2.4 Save model and results
        tdir = os.path.join(OUTPUT_DIR, f"{target}_{MODEL_NAME}")
        os.makedirs(tdir, exist_ok=True)
        joblib.dump(model, os.path.join(tdir, f"{target}_{MODEL_NAME}_model.pkl"))

        # 5.2.5 Validation plots (train vs test)
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        for ax, X, y, label in zip(axs,
                                   [X_train, X_test],
                                   [y_train, y_test],
                                   ['Train', 'Test']):
            y_pred = model.predict(X)
            plot_regression(ax, y, y_pred, cv_r2 if label == 'Test' else None)
            ax.set_title(f"{label} (n={len(y)})")
        plt.tight_layout()
        plt.savefig(os.path.join(tdir, "Validation.png"), dpi=600)
        plt.close()

        # 5.2.6 SHAP analysis ----------------------------------------------------
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(X_train)

        # Composite SHAP dependency plots (4x4 grid)
        fig, axes = plt.subplots(4, 4, figsize=(5, 5))
        axes = axes.flatten()
        for i, f in enumerate(X_train.columns):
            plot_shap_single(axes[i], f, X_train, shap_vals, target)
        # Hide unused sub-plots
        for j in range(len(X_train.columns), len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(tdir, "SHAP_composite.png"), dpi=600)
        plt.close()

        # Horizontal bar plot of mean |SHAP|
        imp = pd.DataFrame({
            'feature': X_train.columns,
            'importance': np.abs(shap_vals.values).mean(axis=0),
            'color': [get_color(f) for f in X_train.columns]
        }).sort_values('importance', ascending=True)

        plt.figure(figsize=(3, 3))
        plt.barh(imp['feature'], imp['importance'], color=imp['color'])
        plt.xlabel("Mean |SHAP value|")
        plt.title("SHAP Summary")
        legend_elements = [Patch(facecolor=v, label=k) for k, v in COLOR_MAP.items()]
        plt.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(tdir, "SHAP_summary.png"), dpi=600)
        plt.close()

        # 5.2.7 Console feedback
        print(f"{target} finished — CV R² = {cv_r2['mean']:.3f}±{cv_r2['std']:.3f}")

# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()