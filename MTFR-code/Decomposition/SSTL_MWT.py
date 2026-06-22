'''
    aSTL-WT for multivariate time series
'''
import sys
import os
import time
import random
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import torch
import gc

from tools import Data_Standarder
from ModelUtils import TimeSeries_Exp
from datetime import datetime
from statsmodels.tsa.seasonal import STL, MSTL
from scipy.linalg import hankel, svd

seed = 42
torch.manual_seed(seed)
np.random.seed(seed) # 0
random.seed(seed)    # 666

current_decompfile_path = os.path.abspath(__file__)
current_decomp_method = "SSTL_MWT"

def wavelet_decomposition_and_reconstruction(signal, wavelet, level):
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 分解项的还原
    reconstructed_components = []
    for i in range(len(coeffs)):
        # 仅保留第 i 个分解项，其他置零
        coeffs_i = [c if j == i else np.zeros_like(c) for j, c in enumerate(coeffs)]
        reconstructed = pywt.waverec(coeffs_i, wavelet)
        reconstructed_components.append(reconstructed[:len(signal)])  # 截断至原信号长度
    reconstructed_components = np.array(reconstructed_components)
        
    return coeffs, reconstructed_components


def WT(data, wavelet, level):
    if data.ndim == 1 or data.shape[1] == 1:
        # data is a single-variable series
        data = data.reshape(-1)
        _, components = wavelet_decomposition_and_reconstruction(data, wavelet, level)
        Components = components.reshape((components.shape[0],1,components.shape[1]))
    else:
        for varId in range(data.shape[1]):
            series = np.copy(data[:,varId])
            _, components = wavelet_decomposition_and_reconstruction(series, wavelet, level)
            components = components.reshape((components.shape[0],1,components.shape[1]))
            if varId == 0:
                Components = np.copy(components)
            else:
                Components = np.concatenate((Components, components), axis = 1)
    return Components
    
def minodd(number):
    number = int(number)
    if number % 2 == 0:
        return number + 1
    else:
        return number

# 将一个序列按照某个长度平均化 --------------------------------------------------
def SSA_reconstruct_topId(time_series, window_size):
    """
    对时间序列进行 SSA 分解
    Args:
        time_series: 时间序列数组
        window_size: 窗口大小，用于构建 Hankel 矩阵
        topR: 提取组分的比例
    Returns:
        Series_reconstructed: 重构后的序列集合，type = pd.Dataframe, shape = (series_length, num)
    """
    
    N = len(time_series)
    
    # 构建 Hankel 矩阵
    hankel_matrix = hankel(time_series[:window_size], time_series[window_size - 1:])
    
    # 奇异值分解
    U, s, V = svd(hankel_matrix)
    
    topId = 1
    component = np.dot(U[:, :topId], np.dot(np.diag(s[:topId]), V[:topId, :]))
    
    L, M = component.shape[0], component.shape[1]
    Lp, Kp = np.min((L, M)), np.max((L, M))
    if L <= M:
        Component_X = component
    else:
        Component_X = component.T

    component_series = []
    for k in range(Lp):
        yi = []
        for m in range(k+1):
            # 第k个(k=0,1,...,Lp-1)有k+1个元素相加
            yi.append(Component_X[m,k-m])
        component_series.append(np.mean(yi))
    for k in range(Kp-Lp):
        yi = []
        for m in range(Lp):
            # 每一部分均有Lp个元素相加, 下标和为Lp
            yi.append(Component_X[m,Lp+k-m])
        component_series.append(np.mean(yi))
    for k in range(N-Kp):
        yi = []
        for m in range(k+1, Lp):
            yi.append(Component_X[m,Kp+k-m])
        component_series.append(np.mean(yi))
    
    return np.array(component_series)


# 使用MWT分解数据
def residual_decomp(residMatrix, decompNum, TargetId = -1, wavelet = "db10"):
    '''

    Parameters
    ----------
    residMatrix : np.adarray or pd.Dataframe
        shape = (time_lag, features)
    decompNum : int
        decomposition number
    TargetId : TYPE, optional
        DESCRIPTION. The default is -1.
        If residMatrix.type is DataFrame,

    Returns
    -------
    resid_decomp : dict
                  {
                   "residual-1": pd.DataFrame,
                   ...
                   "residual-decompNum": pd.DataFrame
                   }

    '''
    if isinstance(residMatrix, pd.DataFrame):
        if type(TargetId) == str:
            targetSeries = residMatrix[TargetId].values
            residMatrix.drop(columns = [TargetId], axis = 1, inplace = True)
            residMatrix[TargetId] = list(targetSeries)                    # 将目标序列放置于最后一列
        elif type(TargetId) == int:
            targetSeries = residMatrix.iloc[:, TargetId]
            Target = residMatrix.columns[TargetId]
            residMatrix.drop(columns = [Target], axis = 1, inplace = True)
            residMatrix[Target] = list(targetSeries)
        resid_cols = list(residMatrix.columns)
        residMatrix = residMatrix.values      
    else:
        resid_cols = []
        for i in range(residMatrix.shape[1] - 1):
            resid_cols.append(f"component-{i+1}")
        resid_cols.append("remainder")
        # residMatrix = residMatrix.T               # (features, time_lag) -> (time_lag, features)
    
    Components = WT(residMatrix, wavelet, decompNum-1)
    residNum = Components.shape[0]
    resid_decomp = {}
    for resid_idx in np.arange(1,1+residNum).astype(int):
        # create each residual dataframe one by one
        resid_df_values = np.zeros_like(residMatrix)
        for col_idx in np.arange(Components.shape[1]):
            resid_df_values[:, col_idx] = np.copy(Components[resid_idx-1, col_idx, :]).reshape(-1)
        resid_df = pd.DataFrame(resid_df_values, columns = resid_cols)
        resid_decomp[f"residual-{resid_idx}"] = resid_df
    
    return resid_decomp
            
    
            
    

# Version I
def SSTL_MWT(data, periods, wavelet, target = -1, resid_num = 1, LOESS_Span = None):
    
    class data_class():
        def __init__(self, observe, trend, seasonals, residual, target):
            self.target = target                # str
            self.observed = observe             # np.ndarray
            self.trend = trend                  # np.ndarray
            self.seasonals = seasonals          # pd.DataFrame
            self.resid = residual               # dict {
                                                #       "residual-1": pd.DataFrame,
                                                #         ...
                                                #       "residual-resid_num": pd.DataFrame
                                                #      }
    
    if type(target) == str:
        series = data[target].values
        data = data.drop(target, axis = 1)
        data[target] = list(series)
    elif type(target) == int:
        target = data.columns[target]
        series = data[target].values
        data = data.drop(target, axis = 1)
        data[target] = list(series)
    
    if LOESS_Span:
        LOESS = max([int(data.shape[0]*LOESS_Span),7])
        if LOESS % 2 == 0:
            LOESS += 1   
        
    if np.ndim(periods) == 0:
        periods = [periods]
    
    if len(periods) == 1:
        # 单季节分解
        period = periods[0]
        if LOESS_Span:
            n_lowpass = max([minodd(3*LOESS_Span),3,minodd(period+1)])
            n_seasonal= max([7, minodd(7*LOESS_Span)])
            n_trend = max([1.5*period/(1-1.5/n_seasonal), 
                           minodd(LOESS_Span*1.5*period/(1-1.5/n_seasonal)),
                           minodd(period+1)])
            stl = STL(
                        series, period = period,
                        seasonal = n_seasonal, 
                        trend = n_trend, 
                        low_pass = n_lowpass
                      ).fit()
        else:
            stl = STL(series, period = period).fit()
        
        # 趋势项
        trend = stl.trend   
        
        # 季节项
        seasonal = SSA_reconstruct_topId(stl.seasonal, period)
        seasonal_dict = {f'seasonal-{period}': seasonal.tolist()}
        seasonals = pd.DataFrame(seasonal_dict)
        
        # 残差项
        remainderM = data.copy()
        remainderM[target] = list(remainderM[target].values - trend - seasonal)
        resid = residual_decomp(residMatrix=remainderM, decompNum=resid_num, TargetId=target, wavelet=wavelet)
        
    else:
        # 多季节分解
        periods = np.sort(np.array(periods))
        periods = tuple(periods)
        
        if LOESS_Span:
            windows_paras = list(minodd(7*LOESS_Span) + 4*np.arange(1, 1+len(periods), 1))
            mstl = MSTL(
                        series, periods=periods,
                        windows = windows_paras
                        ).fit()
        else:
            mstl = MSTL(series, periods=periods).fit()
        
        # 趋势
        trend = mstl.trend
        
        # 季节项
        seasonal_init = mstl.seasonal
        seasonal_dict = {}
        for i in range(seasonal_init.shape[1]):
            seasonal_component = SSA_reconstruct_topId(np.array(seasonal_init[:, i]).reshape(-1),periods[i])
            seasonal_component = seasonal_component.reshape(-1)
            seasonal_dict[f'seasonal-{periods[i]}'] = seasonal_component.tolist()

        seasonals = pd.DataFrame(seasonal_dict)
        
        # 剩余项
        remainderM = data.copy()
        remainderM[target] = list(remainderM[target].values - trend - seasonals.sum(axis = 1))
        resid = residual_decomp(residMatrix=remainderM, decompNum=resid_num, TargetId=target, wavelet=wavelet)
    
    result = data_class(series, trend, seasonals, resid, target)
    
    return result


class decomposer:
    def __init__(self, wavelet):
        self.wavelet = wavelet
    
    def decomp(self, data, periods, target = -1, resid_num = 1, LOESS_Span = None):
        Decomposer = SSTL_MWT(data, periods, self.wavelet, target, resid_num, LOESS_Span)
        self.target = Decomposer.target                # str
        self.observed = Decomposer.observed             # np.ndarray
        self.trend = Decomposer.trend                  # np.ndarray
        self.seasonals = Decomposer.seasonals          # pd.DataFrame
        self.resid = Decomposer.resid    
        return self



def plot_decomp(decomp, fig1_ncols = 2, is_observed = True,
                folder_path = None, plot_suptitle = None,
                data_name = '', fig2_ncols = 0):
    
    if (plot_suptitle is None) and (data_name != ''):
        plot_suptitle = f"{data_name} aSTL-MWT decomposition result"
    elif plot_suptitle is None:
        plot_suptitle = "aSTL-MWT decomposition result"
    
    # Figure 1
    if is_observed:
        if data_name == '':
            data_dict = {'observed': decomp.observed.tolist(),
                         'trend': decomp.trend.tolist()}
        else:
            data_dict = {data_name: decomp.observed.tolist(),
                         'trend': decomp.trend.tolist()}
    else:
        data_dict = {'trend': decomp.trend.tolist()}
    data_df = pd.DataFrame(data_dict)
    data_df = pd.concat([data_df, decomp.seasonals], axis=1)
    for i, (resid_ID, resid_df) in enumerate(decomp.resid.items()):
        Target_df = resid_df[[decomp.target]].rename(columns = {decomp.target: resid_ID})
        if i == 0:
            resid_DF = Target_df.copy()
        else:
            resid_DF = pd.concat([resid_DF,Target_df.copy()], axis = 1)
    data_df = pd.concat([data_df, resid_DF], axis=1)
    
    ncols = fig1_ncols
    nrows = np.ceil(data_df.shape[1]/ncols).astype(int)
    fig1 = plt.figure(figsize=(10*ncols, 3*nrows))
    for i,column in enumerate(data_df.columns):
        plt.subplot(nrows, ncols, i+1)
        plt.plot(data_df[column].values)
        plt.title(column, fontsize = 10*ncols+5, family = 'Century')
        plt.xticks(fontsize = 10*ncols, family = 'Century')
        plt.yticks(fontsize = 10*ncols, family = 'Century')
    
    plt.tight_layout()
    plt.subplots_adjust(top = 0.9)
    plt.suptitle(plot_suptitle, fontsize = 10*ncols+8, y = 1.00, family = 'Century')
    _ = gc.collect()
    
    if folder_path is not None:
        if not os.path.exists(folder_path):
            print(f'The folder path is not exist, and construct the folder ({folder_path}).')
            os.makedirs(folder_path)
        fig1.savefig(f'{folder_path}/{plot_suptitle}.png', dpi = 500)

    
    
def resid_decompNum_Exp(model, data, train_data, args):
    # 处理剩余项数据
    remainder = data.resid  # dict
    train_remainder = train_data.resid
    for i,(resid_key, resid_df) in enumerate(remainder.items()):
        args.input_size = resid_df.shape[1]
        train_resid_df = train_remainder[resid_key].copy()
        
        resid_result = TimeSeries_Exp(resid_df, train_resid_df, model, args, target=data.target)
        resid_pred = resid_result.original_preds
        resid_true = resid_result.original_trues
        if i == 0:
            resid_preds = np.copy(resid_pred)
            resid_trues = np.copy(resid_true)
        else:
            resid_preds += np.copy(resid_pred)
            resid_trues += np.copy(resid_true)
    return resid_trues, resid_preds

