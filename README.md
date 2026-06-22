# MTFR
It is the code for the paper "Multivariable Temporal-Frequency Reconstruction model with two-stage decomposition and explainable AI-driven feature selection for volatile time series forecasting". 

It is a hybrid forecasting model working for multivariable time series forecasting. It integrates the multivariable data decomposition and feature selection based on explainable artificial intelligence (XAI). It can handle multivariable forecasting tasks (multivariate-to-univariate tasks), and is particularly suitable for volatile time series.

The codes and the related datasets will be made publicly available after the paper is accepted.

## Install
The dependency packages of the current repository are as follows:
```
package              | version
----------------------------------------
matplotlib           | 3.7.2
numpy                | 1.23.5
pandas               | 2.0.3
nvidia-ml-py         | 13.580.82
PyWavelets           | 1.4.1
scipy                | 1.10.1
shap                 | 0.44.1
scikit-learn         | 1.3.0
sktime               | 0.29.1
statsmodels          | 0.14.0
torch                | 2.4.1
```
It is not necessary to strictly adhere to the versions of the aforementioned packages; creating an environment where they can operate jointly is sufficient.

## Project context
```
MTFR/
├── Decomposition
│   ├── SSTL_MWT.py
│   └── SSTL_UFFT.py
├── FeatureSelection
│   ├── ALLFeatures.py
│   ├── DeepSHAP.py
│   └── PI.py
├── datasets
│   ├── BJAQ.csv
│   ├── EVCD.csv
│   ├── FDSL.csv
│   ├── GWL.csv
│   └── SILO_Cairns.csv
├── Configs.py
├── MTFR_Pytorch.py
├── ModelUtils.py
├── Models.py
├── main_Exp.py
└── tools.py
```
