# MTFR
It is the code for the paper "Multivariable Temporal-Frequency Reconstruction model with two-stage decomposition and explainable AI-driven feature selection for highly volatile time-series forecasting". 

It is a hybrid forecasting model working for multivariable time series forecasting. It integrates the multivariable data decomposition and feature selection based on explainable artificial intelligence (XAI). It can handle multivariable forecasting tasks (multivariate-to-univariate tasks), and is particularly suitable for volatile time series.



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
It is not necessary to strictly adhere to the versions of the aforementioned packages. Creating an environment where they can operate jointly is sufficient.

## Project context
```
MTFR/
├── main_Exp.py
├── Configs.py
├── MTFR_Pytorch.py
├── ModelUtils.py
├── Models.py
├── tools.py
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
```

## Run
The file `main_Exp.py` serves as the main program. Adjusting the parameters in the `settings` section according to the task scenario is sufficient. Further modifications to the default parameters can be made in the file `Configs.py`.

## Acknowledgement
We appreciate the following GitHub repos a lot for their valuable code base or datasets:

[STL-ALN_BSA-LSTM](https://github.com/zjuml/STL-ALNBSA-LSTM)

[CEEMDAN_LSTM](https://github.com/FateMurphy/CEEMDAN_LSTM)

[Time-Series-Library](https://github.com/thuml/Time-Series-Library)


