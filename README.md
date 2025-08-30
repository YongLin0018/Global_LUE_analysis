# Global_LUE_analysis
This repository contains Python code for modeling and mapping global Light Use Efficiency (LUE) using machine learning and spatial analysis techniques. 

Python Requirements:
Python 3.10+
Required packages: pip install pandas numpy scipy matplotlib seaborn scikit-learn shap rasterio joblib openpyxl statsmodels

The files we provide include:
1. ridge analysis.py is for the Ridge-regression based sensitivity analysis for Light-Use-Efficiency (LUE)
2. SHAP analysis.py is for Light-Use-Efficiency (LUE) modelling and interpretation pipeline
3. Mapping global LUE.py is for spatial scaling of Light Use Efficiency (LUE) using Random Forest modeling
4. example data for ridge or SHAP analysis.xlsx is the example data for ridge analysis.py and SHAP analysis.py
5. example data for upscaling.xlsx is the example data for Mapping global LUE.py

The output including visualization results of ridge and SHAP analysis, as well as predicted LUE with a resolution of 0.05 °.
In order to run Mapping global LUE.py, a folder containing tif files of 16 prediction parameters with a resolution of 0.05 ° needs to be prepared.

Contact
​​Yong Lin​​
Institute of Geographic Sciences and Natural Resources Research, CAS
Email: linyong0018@igsnrr.ac.cn
