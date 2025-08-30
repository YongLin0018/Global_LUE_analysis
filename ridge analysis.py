"""
Code for ridge analysis
Author: Yong Lin
Email: linyong0018@igsnrr.ac.cn
Institution: Institute of Geographic Sciences and Natural Resources Research, CAS
Date: 2025-08-30

Description:
    Ridge-regression based sensitivity analysis for Light-Use-Efficiency (LUE)
    across three targets (LUEmax, LUEinc, LUEact).  The script:
    1. Standardises predictors and fits Ridge-CV models.
    2. Computes coefficient uncertainty via the ridge-covariance matrix.
    3. Generates two-panel figures:
        - Left: absolute coefficient effects ± 1 σ (sorted).
        - Right: grouped contribution (%) to total explained variance,
          with Canopy sub-groups (Structure vs. Physiology) stacked.
    4. Saves PNG/PDF figures and a CSV of coefficients to a time-stamped folder.
"""

# ───────────────────────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from matplotlib.patches import Patch

# ───────────────────────────────────────────────────────────────────────────────
# Plot configuration (global)
# ───────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 5,
    'axes.titlesize': 5,
    'axes.labelsize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'figure.dpi': 600
})

# ───────────────────────────────────────────────────────────────────────────────
# Feature grouping dictionaries
# ───────────────────────────────────────────────────────────────────────────────
GROUP_CONFIG = {
    'Climate': ['FD', 'TA', 'SM', 'P', 'VPD'],
    'Canopy structure': ['FPAR', 'MLA', 'SLA'],
    'Canopy physiology': ['CHL', 'GSL', 'LN', 'LP'],
    'Nutrient': ['SOC', 'SN', 'SP', 'CNR']
}

# Merge Canopy sub-groups for the inset bar plot
BIG_GROUP_CONFIG = {
    'Climate': ['FD', 'TA', 'SM', 'P', 'VPD'],
    'Canopy': GROUP_CONFIG['Canopy structure'] + GROUP_CONFIG['Canopy physiology'],
    'Nutrient': ['SOC', 'SN', 'SP', 'CNR']
}

# Colours
CANOPY_SUBGROUP_COLORS = {
    'Canopy structure': '#31553B',
    'Canopy physiology': '#C7C51B'
}

COLOR_MAP = {
    'Climate': '#74CEF7',
    'Nutrient': '#7072A6',
    'Canopy': (
        CANOPY_SUBGROUP_COLORS['Canopy structure'],
        CANOPY_SUBGROUP_COLORS['Canopy physiology']
    )
}

# ───────────────────────────────────────────────────────────────────────────────
# Core analysis function
# ───────────────────────────────────────────────────────────────────────────────
def ridge_analysis(data: pd.DataFrame, targets: list[str]) -> None:
    """Run ridge regression for each target and generate figures."""
    base_features = [c for c in data.columns if c not in targets]

    for target in targets:
        print(f"\nProcessing target: {target}")
        # Select only features present in the grouping dictionaries
        features = [f for f in base_features
                    if any(f in grp for grp in sum(GROUP_CONFIG.values(), []))]

        # Standardise and fit
        X_std = StandardScaler().fit_transform(data[features])
        y = data[target].values

        ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
        ridge.fit(X_std, y)
        y_pred = ridge.predict(X_std)
        r2 = r2_score(y, y_pred)

        # ── Coefficient uncertainty via ridge-covariance matrix ──
        residuals = y - y_pred
        sigma2 = np.sum(residuals ** 2) / (len(y) - len(features))
        XtX = X_std.T @ X_std
        alpha = ridge.alpha_
        p = X_std.shape[1]
        ridge_cov = (
            sigma2 * np.linalg.inv(XtX + alpha * np.eye(p))
            @ XtX @ np.linalg.inv(XtX + alpha * np.eye(p))
        )
        coef_std = np.sqrt(np.diag(ridge_cov))

        # ── Main figure --------------------------------------------------------
        fig = plt.figure(figsize=(2.3, 2.3))
        gs  = fig.add_gridspec(1, 1)
        ax  = fig.add_subplot(gs[0])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        coefs = pd.Series(ridge.coef_, index=features)
        stds  = pd.Series(coef_std, index=features)

        # Sort by absolute magnitude
        sort_idx = np.argsort(-np.abs(coefs))
        coefs    = coefs.iloc[sort_idx]
        stds     = stds.iloc[sort_idx]

        # Assign colors
        colors = []
        for feat in coefs.index:
            group = next((k for k, v in GROUP_CONFIG.items() if feat in v), 'Other')
            colors.append(
                COLOR_MAP[group] if group not in ['Canopy structure', 'Canopy physiology']
                else CANOPY_SUBGROUP_COLORS[group]
            )

        y_pos = np.arange(len(coefs))
        for i, (coef, std, color) in enumerate(zip(coefs, stds, colors)):
            abs_coef = abs(coef)
            ax.scatter(abs_coef, i, color=color, s=15, zorder=3)
            ax.hlines(i, max(0, abs_coef - std), abs_coef + std,
                      color='gray', linewidth=0.5, zorder=2)
            ax.vlines(max(0, abs_coef - std), i - 0.15, i + 0.15,
                      color='gray', linewidth=0.5, zorder=2)
            ax.vlines(abs_coef + std, i - 0.15, i + 0.15,
                      color='gray', linewidth=0.5, zorder=2)

            sign = '+' if coef >= 0 else '-'
            ax.text(abs_coef, i, sign, ha='center', va='center',
                    color='white', fontsize=5, fontweight='bold', zorder=4)

        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_title(f'{target} (R² = {r2:.2f})', pad=5)
        ax.set_xlabel('Absolute Effect')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(coefs.index)
        ax.set_ylim(-0.5, len(coefs) - 0.5)
        ax.invert_yaxis()

        # ── Inset: grouped contribution ----------------------------------------
        inset_ax = ax.inset_axes([0.55, 0.15, 0.4, 0.35])
        for spine in inset_ax.spines.values():
            spine.set_linewidth(0.5)

        # Compute aggregated effects and uncertainties
        group_effects = {}
        group_stds    = {}
        canopy_subgroup_effects = {}
        canopy_subgroup_stds    = {}

        # Canopy sub-groups
        for group, vars in GROUP_CONFIG.items():
            if 'Canopy' in group:
                g_feats = [f for f in features if f in vars]
                if not g_feats:
                    eff = std = 0
                else:
                    eff = coefs[g_feats].abs().sum()
                    idx = [features.index(f) for f in g_feats]
                    var = ridge_cov[np.ix_(idx, idx)].sum()
                    std = np.sqrt(var)
                canopy_subgroup_effects[group] = eff
                canopy_subgroup_stds[group]    = std

        # Big groups (Climate, Canopy, Nutrient)
        for group, vars in BIG_GROUP_CONFIG.items():
            g_feats = [f for f in features if f in vars]
            if not g_feats:
                eff = std = 0
            else:
                eff = coefs[g_feats].abs().sum()
                idx = [features.index(f) for f in g_feats]
                var = ridge_cov[np.ix_(idx, idx)].sum()
                std = np.sqrt(var)
            group_effects[group] = eff
            group_stds[group]    = std

        # Convert to percentages
        total_eff = sum(group_effects.values())
        group_ratios = {k: v / total_eff * 100 for k, v in group_effects.items()}
        group_stds   = {k: v / total_eff * 100 for k, v in group_stds.items()}

        canopy_structure_ratio = (
            canopy_subgroup_effects['Canopy structure'] / total_eff * 100
        )
        canopy_physiology_ratio = (
            canopy_subgroup_effects['Canopy physiology'] / total_eff * 100
        )

        # Prepare bar data
        group_df = pd.Series(group_ratios).sort_values(ascending=False)
        group_std = pd.Series(group_stds).reindex(group_df.index)
        colors = [COLOR_MAP[k] for k in group_df.index]

        # Plot horizontal bars
        y_pos = np.arange(len(group_df))
        bh = 0.6
        eb = 0.15  # error-bar cap width
        for i, (grp, ratio) in enumerate(group_df.items()):
            if grp == 'Canopy':
                # Stacked bars for Canopy sub-groups
                b_struct = inset_ax.barh(
                    i, canopy_structure_ratio, height=bh,
                    left=0, color=CANOPY_SUBGROUP_COLORS['Canopy structure'],
                    edgecolor='none', zorder=2
                )
                b_physio = inset_ax.barh(
                    i, canopy_physiology_ratio, height=bh,
                    left=canopy_structure_ratio,
                    color=CANOPY_SUBGROUP_COLORS['Canopy physiology'],
                    edgecolor='none', zorder=2
                )
            else:
                b = inset_ax.barh(
                    i, ratio, height=bh,
                    color=COLOR_MAP[grp], edgecolor='none', zorder=2
                )

            # Error bars
            inset_ax.hlines(i,
                            max(0, ratio - group_std[grp]),
                            ratio + group_std[grp],
                            colors='gray', linewidth=0.5, zorder=3)
            inset_ax.vlines(max(0, ratio - group_std[grp]),
                            i - eb, i + eb, colors='gray', linewidth=0.5, zorder=3)
            inset_ax.vlines(ratio + group_std[grp],
                            i - eb, i + eb, colors='gray', linewidth=0.5, zorder=3)

        # Inset cosmetics
        inset_ax.set_xlabel('Contribution (%)', labelpad=0)
        inset_ax.set_yticks(y_pos)
        inset_ax.set_yticklabels(group_df.index, fontsize=5)
        inset_ax.set_xlim(0, 100)
        inset_ax.grid(False)
        inset_ax.set_axisbelow(True)

        # Sub-legend for Canopy
        legend_handles = [
            Patch(facecolor=CANOPY_SUBGROUP_COLORS['Canopy structure'],
                  label='Structure', linewidth=0.5),
            Patch(facecolor=CANOPY_SUBGROUP_COLORS['Canopy physiology'],
                  label='Physiology', linewidth=0.5)
        ]
        inset_ax.legend(handles=legend_handles, loc='upper left',
                        fontsize=5, frameon=False,
                        bbox_to_anchor=(0.18, 1.05), handlelength=1, handleheight=1)

        # ── Save outputs -------------------------------------------------------
        plt.savefig(os.path.join(output_folder, f"{target}_effects.png"),
                    bbox_inches='tight', pad_inches=0.1)
        plt.savefig(os.path.join(output_folder, f"{target}_effects.pdf"),
                    bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Save coefficients to CSV
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': ridge.coef_,
            'Std': coef_std,
            'Group': [next((k for k, v in GROUP_CONFIG.items() if f in v), 'Other')
                      for f in features],
            'R2': r2
        }).sort_values('Coefficient', key=np.abs, ascending=False)
        coef_df.to_csv(os.path.join(output_folder, f"{target}_coefficients.csv"),
                       index=False)

# ───────────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    output_folder = os.path.join(f"LUE_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_folder, exist_ok=True)

    # Load data  (replace with your actual path)
    data_path = r"example data for ridge or SHAP analysis.xlsx"
    data = pd.read_excel(data_path)

    # Run
    ridge_analysis(data, targets=['LUEmax', 'LUEinc', 'LUEact'])

    print(f"\nAll results saved to: {output_folder}")