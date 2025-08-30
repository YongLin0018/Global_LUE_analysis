"""
Code for mapping global LUE
Author: Yong Lin
Email: linyong0018@igsnrr.ac.cn
Institution: Institute of Geographic Sciences and Natural Resources Research, CAS
Date: 2025-08-30

Description:
This script performs spatial scaling of Light Use Efficiency (LUE) parameters using:
1. Random Forest regression modeling
2. SHAP value interpretation for feature importance
3. Global prediction mapping with uncertainty quantification

Dependencies:
Python 3.10+, pandas, matplotlib, scikit-learn, shap, seaborn, rasterio, joblib, numpy, scipy
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import shap
import numpy as np
from scipy import stats
import seaborn as sns
import rasterio
from rasterio.windows import Window
import joblib
from matplotlib.patches import Patch

# --------------------------------------------------
# 0. 计算脚本所在目录，保证路径绝对且可移植
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================
# CONFIGURATION（用户只需改这里）
# ================================
INPUT_DATA_FILE   = os.path.join(SCRIPT_DIR, "example data for upscaling.xlsx")  # Excel 数据
OUTPUT_DIRECTORY  = os.path.join(SCRIPT_DIR, "LUE_Upscaling_Results")            # 结果输出
TIF_DIR           = os.path.join(SCRIPT_DIR, "TIF")                              # 栅格目录

# 其余配置保持不变
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

TARGETS = ['LUEmax', 'LUEinc', 'LUEact']
MODEL_NAME  = 'RF'
MODEL_CLASS = RandomForestRegressor
# ================================

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
plt.rcParams.update({'font.size': 6, 'figure.figsize': (3, 3)})


class CustomRFWrapper(MODEL_CLASS):
    """Custom Random Forest wrapper preserving feature names"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_names_ = None

    def fit(self, X, y, **fit_params):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        return super().fit(X, y, **fit_params)


def get_feature_color(feature_name):
    """Return color for feature based on its group"""
    for group, features in GROUP_CONFIG.items():
        if feature_name in features:
            return COLOR_MAP[group]
    return '#999999'


def plot_regression(y_true, y_pred, dataset_type, save_path):
    """Generate validation scatter plot"""
    plt.figure(figsize=(3, 3))
    r2 = r2_score(y_true, y_pred)
    slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
    sns.regplot(x=y_true, y=y_pred,
                scatter_kws={'s': 15, 'color': 'blue', 'alpha': 0.5, 'edgecolor': 'none'},
                line_kws={'color': 'red', 'lw': 1})
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=1)
    plt.text(0.05, 0.85, f"R² = {r2:.2f}\ny = {slope:.2f}x + {intercept:.2f}",
             transform=plt.gca().transAxes, ha='left', va='top')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title(f"{MODEL_NAME} - {dataset_type}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()


def shap_analysis(model, X_train, y_train, X_test, y_test, target_var):
    """SHAP 分析与可视化"""
    model.fit(X_train, y_train)
    target_dir = os.path.join(OUTPUT_DIRECTORY, f"{target_var}_{MODEL_NAME}")
    os.makedirs(target_dir, exist_ok=True)
    joblib.dump(model, os.path.join(target_dir, f"{target_var}_model.pkl"))

    # 验证图
    for dataset, X, y in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        y_pred = model.predict(X)
        plot_regression(y, y_pred, dataset,
                        os.path.join(target_dir, f"Validation_{dataset}.png"))

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)

    # 特征重要性
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': np.abs(shap_values.values).mean(axis=0),
        'Color': [get_feature_color(f) for f in X_train.columns]
    }).sort_values('Importance', ascending=True)

    plt.figure(figsize=(3, 3))
    plt.barh(importance_df['Feature'], importance_df['Importance'],
             color=importance_df['Color'])
    legend_items = [Patch(facecolor=v, label=k) for k, v in COLOR_MAP.items()]
    plt.legend(handles=legend_items, loc='lower right')
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "SHAP_Feature_Importance.png"), dpi=600)
    plt.close()

    # 单变量依赖图
    for feature in X_train.columns:
        idx = list(X_train.columns).index(feature)
        plt.figure(figsize=(2, 2))
        sns.regplot(x=X_train[feature], y=shap_values.values[:, idx],
                    scatter_kws={'s': 6, 'color': get_feature_color(feature), 'alpha': 0.5},
                    line_kws={'color': 'black', 'lw': 1}, lowess=True)
        corr, pval = stats.pearsonr(X_train[feature], shap_values.values[:, idx])
        plt.annotate(f"r = {corr:.2f}\np = {pval:.2e}",
                     xy=(0.05, 0.9), xycoords='axes fraction', ha='left', va='top', fontsize=6)
        plt.title(f"SHAP: {feature}")
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, f"SHAP_{feature}.png"), dpi=600)
        plt.close()


def generate_global_predictions():
    """生成全球预测栅格（含不确定性）"""
    # 统一使用绝对路径
    RASTER_FILES = {
        'GSL':  os.path.join(TIF_DIR, "GSL.tif"),
        'FPAR': os.path.join(TIF_DIR, "FPAR.tif"),
        'FD':   os.path.join(TIF_DIR, "FD.tif"),
        'TA':   os.path.join(TIF_DIR, "TA.tif"),
        'SM':   os.path.join(TIF_DIR, "SM.tif"),
        'P':    os.path.join(TIF_DIR, "P.tif"),
        'VPD':  os.path.join(TIF_DIR, "VPD.tif"),
        'SOC':  os.path.join(TIF_DIR, "SOC.tif"),
        'SN':   os.path.join(TIF_DIR, "SN.tif"),
        'CNR':  os.path.join(TIF_DIR, "CNR.tif"),
        'MLA':  os.path.join(TIF_DIR, "MLA.tif"),
        'CHL':  os.path.join(TIF_DIR, "CHL.tif"),
        'SLA':  os.path.join(TIF_DIR, "SLA.tif"),
        'LN':   os.path.join(TIF_DIR, "LN.tif"),
        'LP':   os.path.join(TIF_DIR, "LP.tif"),
        'SP':   os.path.join(TIF_DIR, "SP.tif")
    }

    output_dir = os.path.join(OUTPUT_DIRECTORY, "Global_Predictions")
    os.makedirs(output_dir, exist_ok=True)

    # 参考栅格
    ref_path = RASTER_FILES['GSL']
    with rasterio.open(ref_path) as src:
        profile = src.profile
        height, width = src.shape
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

    BLOCK_SIZE = 256

    for target in TARGETS:
        model_path = os.path.join(OUTPUT_DIRECTORY, f"{target}_{MODEL_NAME}", f"{target}_model.pkl")
        if not os.path.exists(model_path):
            continue
        model = joblib.load(model_path)
        if not hasattr(model, 'feature_names_'):
            continue

        pred_path   = os.path.join(output_dir, f"{target}_Prediction.tif")
        uncert_path = os.path.join(output_dir, f"{target}_Uncertainty.tif")

        with rasterio.open(pred_path, 'w', **profile) as pred_dst, \
             rasterio.open(uncert_path, 'w', **profile) as uncert_dst:

            for y in range(0, height, BLOCK_SIZE):
                for x in range(0, width, BLOCK_SIZE):
                    win_w = min(BLOCK_SIZE, width - x)
                    win_h = min(BLOCK_SIZE, height - y)
                    window = Window(x, y, win_w, win_h)

                    # 读取特征块
                    X_block = []
                    for feat in model.feature_names_:
                        with rasterio.open(RASTER_FILES[feat]) as src:
                            data = src.read(1, window=window, boundless=True, fill_value=np.nan)
                            data = data.astype(np.float32)
                            data[data == src.nodata] = np.nan
                            X_block.append(data.flatten())

                    X = np.column_stack(X_block)
                    valid = ~np.isnan(X).any(axis=1)

                    pred = np.full(X.shape[0], np.nan, dtype=np.float32)
                    unc  = np.full(X.shape[0], np.nan, dtype=np.float32)

                    if np.any(valid):
                        valid_X = pd.DataFrame(X[valid], columns=model.feature_names_)
                        tree_preds = np.array([tree.predict(valid_X) for tree in model.estimators_])
                        pred[valid] = tree_preds.mean(axis=0)
                        unc[valid]  = tree_preds.std(axis=0)

                    pred_dst.write(pred.reshape(win_h, win_w), 1, window=window)
                    uncert_dst.write(unc.reshape(win_h, win_w), 1, window=window)


if __name__ == "__main__":
    print("Starting analysis...")

    # 1. 读入数据
    df = pd.read_excel(INPUT_DATA_FILE).dropna()

    # 2. SHAP 建模
    for target_var in TARGETS:
        print(f"\nProcessing target variable: {target_var}")
        features = []
        for group in GROUP_CONFIG.values():
            features += [f for f in group if f in df.columns]
        X = df[features]
        y = df[target_var]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = CustomRFWrapper(n_estimators=500, oob_score=True, random_state=42)
        shap_analysis(model, X_train, y_train, X_test, y_test, target_var)

    # 3. 全球预测
    print("\nGenerating global prediction maps...")
    generate_global_predictions()

    print(f"\nAnalysis complete. Results saved to: {OUTPUT_DIRECTORY}")