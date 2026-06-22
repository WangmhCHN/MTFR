import numpy as np
import pandas as pd
import os
import sys
import gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pynvml, threading, time
import shutil

from sklearn.metrics import *
from sktime.performance_metrics.forecasting import *
from sklearn.preprocessing import StandardScaler
from ModelUtils import *
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, TensorDataset
from Models import obtain_predictor

seed = 42
np.random.seed(seed) # 0
torch.manual_seed(seed)

# 将数据分类训练集的样本与目标 --------------------------------------------------
# 将np.ndarray数据转换为带标签的形式
def arr_loader(Array, pred_len, seq_len):
    '''
        return shape = (length, time_lag, features)
    '''
    X, Y = [], []
    arr_len = Array.shape[0]
    if Array.ndim == 2:
        TargetArray = np.copy(Array[:,-1])
    else:
        TargetArray = np.copy(Array)
    for i in range(0,arr_len-pred_len-seq_len):
        x = np.array(Array[i:i+seq_len])
        y = np.array(TargetArray[i+seq_len:i+seq_len+pred_len])
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    while X.ndim < 3:
        X = np.expand_dims(X, axis = -1)
    while Y.ndim < 3:
        Y = np.expand_dims(Y, axis = -1)
    
    return X, Y


# 处理DataFrame数据转成带标签的形式
def df_loader(df, target, pred_len, seq_len):
    # 处理将目标特征放在最后一列
    TargetSeries = df[target].values
    df = df.drop(target, axis = 1)
    df[target] = list(TargetSeries)
    
    # 获取数据
    DFValues = df.values
    X, Y = arr_loader(DFValues, pred_len, seq_len)
    
    return X, Y
    

# 综合两者
def XYData_Loader(data, pred_len, seq_len, target = -1):
    if isinstance(data, np.ndarray):
        X, Y = arr_loader(data, pred_len, seq_len)
    if isinstance(data, pd.DataFrame):
        if type(target) == int:
            target = data.columns[target]
        X, Y = df_loader(data, target, pred_len, seq_len)
    Y = Y[...,-1]
    return X, Y


# 数据误差计算 -----------------------------------------------------------------
# data loss 
class data_loss:
    '''
        class attribution:
            self.rmse = rmse
            self.mse = mse
            self.mae = mae
            self.mape = mape
            self.mdape = mdape
            self.r2 = r2
            self.ia = ia
            self.tic = tic
            self.corr_coef = corr_coef
            self.max_error = max_error
            self.evs = evs
            self.error_std = error_std
            self.smape = smape
            self.mase = mase 
    '''
    def __init__(self, true_arr, pred_arr, mase_season=1):
        true_arr, pred_arr = np.array(true_arr), np.array(pred_arr)
        while true_arr.ndim < 3:
            true_arr = np.expand_dims(true_arr, axis = -1)
        while pred_arr.ndim < 3:
            pred_arr = np.expand_dims(pred_arr, axis = -1)
        
        if true_arr.shape != pred_arr.shape:
            raise ValueError("true_array and pred_array must have the same shape!")
        self.true_arr = true_arr
        self.pred_arr = pred_arr
        self.true_arr_2D = true_arr.reshape(true_arr.shape[0], -1)
        self.pred_arr_2D = pred_arr.reshape(pred_arr.shape[0], -1)
        self.mase_season = mase_season
        self.epsilon = np.finfo(np.float64).eps
        self._LossCalcalate()
    
    def _LossCalcalate(self):
        self.mse = mean_squared_error(self.true_arr_2D, self.pred_arr_2D)
        self.rmse = np.sqrt(self.mse)
        self.mae = mean_absolute_error(self.true_arr_2D, self.pred_arr_2D)
        self.mape = mean_absolute_percentage_error(self.true_arr_2D, self.pred_arr_2D)
        self.mdape = median_absolute_percentage_error(self.true_arr_2D, self.pred_arr_2D)
        self.r2 = r2_score(self.true_arr_2D, self.pred_arr_2D)
        self.evs = explained_variance_score(self.true_arr_2D, self.pred_arr_2D)

        # Pearson Correlation Coefficient (Corr_coef)
        r_scores = []
        for i in range(self.true_arr_2D.shape[1]):
            r_score = np.corrcoef(self.true_arr_2D[:,i], self.pred_arr_2D[:,i])[1,0]
            r_scores.append(r_score)
        self.corr_coef = np.mean(r_scores)

        # Willmott’s Index of Agreement (IA)
        ia_numerator = np.sum((self.true_arr_2D - self.pred_arr_2D)**2)
        ia_denominator = (np.sum(np.abs(self.true_arr_2D - np.mean(self.true_arr_2D)) + np.abs(self.pred_arr_2D - np.mean(self.pred_arr_2D))))**2
        self.ia = 1 - ia_numerator / ia_denominator

        # Theil’s Inequality Coefficient (TIC)
        tic_numerator = np.sqrt(np.mean((self.true_arr_2D - self.pred_arr_2D)**2))
        tic_denominator = np.sqrt(np.mean(self.true_arr_2D**2)) + np.sqrt(np.mean(self.pred_arr_2D**2))
        self.tic = tic_numerator / tic_denominator

        # Maximum Error
        self.max_error = np.max(np.abs(self.true_arr_2D - self.pred_arr_2D))

        # Standard Deviation of Errors
        self.error_std = np.std(self.true_arr_2D - self.pred_arr_2D)


        # Symmetric Mean Absolute Percentage Error (SMAPE)
        self.smape = np.mean([np.abs(self.true_arr_2D[:,i] - self.pred_arr_2D[:,i]) / np.maximum((np.abs(self.true_arr_2D[:,i]) + np.abs(self.pred_arr_2D[:,i])) / 2, self.epsilon)
                          for i in range(self.true_arr_2D.shape[1])])

        # Mean Absolute Scaled Error (MASE)
        self.mase = np.mean([(np.abs(self.true_arr_2D[:,i] - self.pred_arr_2D[:,i]))/np.mean(np.abs(self.true_arr_2D[:-self.mase_season, i] - self.true_arr_2D[self.mase_season:,i])) 
                for i in range(self.true_arr_2D.shape[1])])
        
        # combined accuracy (DOI:https://doi.org/10.1016/j.jhydrol.2019.123981)
        self.CA = 1/3*(self.rmse+self.mae+(1-self.r2))


def loss_print(data_loss, metrics = ["RMSE", "MAE", "MASE", "R2", "R"], FirstCharacter = "	"):
    if type(metrics) == str:
        metrics = metrics.split(",")
    MetricsMaps = {
                    "RMSE": data_loss.rmse,
                    "MSE": data_loss.mse,
                    "MAE": data_loss.mae,
                    "MAPE": data_loss.mape,
                    "MDAPE": data_loss.mdape,
                    "R2": data_loss.r2,
                    "IA": data_loss.ia,
                    "TIC": data_loss.tic,
                    "R": data_loss.corr_coef,
                    "MAX_ERROR": data_loss.max_error,
                    "EVS": data_loss.evs,
                    "SDE": data_loss.error_std,
                    "SMAPE": data_loss.smape,
                    "MASE": data_loss.mase
                    }
    for i,metric in enumerate(metrics):
        metric = metric.upper()
        if i == 0:
            metric_value = MetricsMaps[metric]
            if metric == "IA":
                print(f"{FirstCharacter}{metric.upper()}: {metric_value:.6f}", end = " ")
            else:
                print(f"{FirstCharacter}{metric.upper()}: {metric_value:.4f}", end = " ")
        else:
            metric_value = MetricsMaps[metric]
            if metric == "IA":
                print(f"| {metric.upper()}: {metric_value:.6f}", end = " ")
            else:
                print(f"| {metric.upper()}: {metric_value:.4f}", end = " ")


# 将当前代码保存在指定文件夹 ----------------------------------------------------  
def save_current_code_to_folder(folder_path, output_filename, cover = False, is_print = True):
    # 获取当前脚本的文件名
    current_script = os.path.abspath(sys.argv[0])

    try:
        # 读取当前脚本的内容
        with open(current_script, 'r', encoding='utf-8') as file:
            code_content = file.read()

        # 确保目标文件夹存在
        os.makedirs(folder_path, exist_ok=True)

        # 构建目标文件的完整路径
        output_path = f'{folder_path}/{output_filename}'
        if os.path.exists(output_path) and cover is False:
            output_name, output_extension = os.path.splitext(output_filename)
            count = 0
            files_in_folder = os.listdir(folder_path)
            for file in files_in_folder:
                file_name, file_extension = os.path.splitext(file)
                if file_extension == output_extension and file_name.startswith(output_name):
                    count += 1
            output_path = f'{folder_path}/{output_name}_{count}{output_extension}'
                    

        # 将内容写入指定文件
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(code_content)
        
        if is_print:
            print(f"Current code saved to {output_path}")
        
        return output_path
    except Exception as e:
        print(f"An error occurred: {e}")    
        
        return None
        


# 画特征重要性图 ---------------------------------------------------------------
def generate_color(num, topId = 0):
    if num == topId:
        return "#0000FF"
    else:
        # 将数字映射到 RGB 通道，范围在 [0, 255] 内
        r = (num * 50) % 256
        g = (num * 100) % 256
        b = (num * 150) % 128
        # 转换成十六进制颜色
        return f"#{r:02X}{g:02X}{b:02X}"

def plot_FeatureImportance(Features_df, topId = 0, plot_title = ""):
    if "Feature" in list(Features_df.columns):
        Features_df = Features_df.set_index("Feature", drop = True)
        # Features_df.drop("Feature", axis = 1, inplace = True)
    if "Class" not in list(Features_df.columns):
        Features_df["Class"] = np.ones((Features_df.shape[0],)).tolist()
    if "claster" in list(Features_df.columns) and "Class" not in list(Features_df.columns):
        Features_df = Features_df.rename(columns = {"claster": "Class"})
    
    topId = int(topId)
    
    fig_height = np.min([2*len(Features_df), 2**15/500]).astype(int)
    fig = plt.figure(figsize=(10, fig_height))
    ax = fig.add_subplot(111)
    
    bars = []
    for i, (feature, importance, class_name) in enumerate(zip(Features_df.index, Features_df['Importance'], Features_df['Class'])):
        color = generate_color(int(class_name), topId)
        bar = ax.barh(feature, importance, color=color)
        bars.append(bar)
    
    # 设置标签和标题
    ax.set_xlabel('Importance', family = "cmr10")  # 设置x轴标签
    ax.set_ylabel('Feature', family = "cmr10")  # 设置y轴标签
    ax.set_title(f'{plot_title}Feature Importance by Class', family = "cmr10")  # 设置图形标题
    ax.invert_yaxis()  # 反转y轴，使得最重要的特征在顶部

    # 在每个条形上添加数值标签
    for bar in bars:
        for b in bar:
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2,  # 文字位置
                    f'{b.get_width():.3f}',  # 文字内容，保留三位小数
                    va='center', ha='left', family = "cmr10")  # 文字垂直居中，水平左对齐
    return fig        
        
        

# def plot_decomp(decomp, ncols = 1, is_observed = True, target_name = None, plot_title = None,
#                 folder_path = None):
#     if target_name is None:
#         target_name = "observed"
#     if is_observed:
#         data_dict = {target_name: list(decomp.observed),
#                      "trend": list(decomp.trend)}
#         data_df = pd.DataFrame(data_dict)
#         data_df = pd.concat([data_df, decomp.seasonals], axis = 1)
#     else:
#         data_dict = {"trend": list(decomp.trend)}
#         data_df = pd.DataFrame(data_dict)
#         data_df = pd.concat([data_df, decomp.seasonals], axis = 1)
#     # extract resid
#     resids_dict = {}
#     for componentId, resid_df in decomp.resid.items():
#         resids_dict[componentId] = resid_df.iloc[:,-1].values.tolist()
#     resids_df = pd.DataFrame(resids_dict)
#     data_df = pd.concat([data_df, resids_df], axis = 1)
    
#     plotNum = data_df.shape[1]
#     nrows = np.ceil(plotNum/ncols).astype(int)
#     fig = plt.figure(figsize = (ncols*5, nrows*3))
#     if plot_title is None:
#         fig.suptitle("decomposition result", family = "Times New Roman", y=1)
#     else:
#         fig.suptitle(plot_title, family = "Times New Roman", y=1)
    
#     x = np.arange(1, 1+data_df.shape[0]).astype(int)
#     for i, componentId in enumerate(data_df.columns):
#         ax = fig.add_subplot(nrows, ncols, i+1)
#         series = data_df[componentId].values
#         ax.plot(x, series, label = componentId)
#         ax.set_title(componentId, family = "Times New Roman")
#         ax.set_xlim([0,data_df.shape[0]+1])
#     fig.tight_layout()
    
#     if folder_path != None:
#         fig.savefig(f"{folder_path}/decomposition result.png")
#         fig.savefig(f"{folder_path}/decomposition result.pdf")
    
#     return fig


    
def plot_decomp(decomp, fig1_ncols = 2, is_observed = True,
                folder_path = None, plot_suptitle = None,
                data_name = '', fig2_ncols = 0, decomp_method = None):
    
    if decomp_method is None:
        if (plot_suptitle is None) and (data_name != ''):
            plot_suptitle = f"{data_name} decomposition result"
        elif plot_suptitle is None:
            plot_suptitle = "decomposition result"
    else:
        if (plot_suptitle is None) and (data_name != ''):
            plot_suptitle = f"{data_name} {decomp_method} decomposition result"
        elif plot_suptitle is None:
            plot_suptitle = "{decomp_method} decomposition result"
        
    
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
    x = np.arange(1, 1+data_df.shape[0]).astype(int)
    fig1 = plt.figure(figsize=(10*ncols, 3*nrows))
    plt.suptitle(plot_suptitle, fontsize = 10*ncols+8, y = 1.00, family = 'cmr10')
    for i,column in enumerate(data_df.columns):
        plt.subplot(nrows, ncols, i+1)
        plt.plot(x, data_df[column].values)
        plt.title(column, fontsize = 10*ncols+5, family = 'cmr10')
        plt.xticks(fontsize = 10*ncols, family = 'cmr10')
        plt.yticks(fontsize = 10*ncols, family = 'cmr10')
        plt.xlim([0, data_df.shape[0]+1])
    
    plt.tight_layout()
    # plt.subplots_adjust(top = 0.9)
    _ = gc.collect()
    
    if folder_path is not None:
        if not os.path.exists(folder_path):
            print(f'The folder path is not exist, and construct the folder ({folder_path}).')
            os.makedirs(folder_path)
        fig1.savefig(f'{folder_path}/{plot_suptitle}.png')
        fig1.savefig(f'{folder_path}/{plot_suptitle}.pdf')
    
    
    
    # Figure 2 (resid decomposition results)
    if fig2_ncols != 0:
        ncols = fig2_ncols
        for idx, (key, component) in enumerate(decomp.resid.items()):
            nrows = np.ceil(component.shape[1]/ncols).astype(int)
            plt.figure(figsize=(10*ncols, 3*nrows))
            plt.suptitle(f'{data_name} resid decomposition result (components-{idx+1})', 
                          fontsize = 10*ncols+8, y = 1.00, family = 'cmr10')
            for i,column in enumerate(component.columns):
                plt.subplot(nrows, ncols, i+1)
                plt.plot(component[column].values)
                plt.title(column, fontsize = 10*ncols+5, family = 'cmr10')
                plt.xticks(fontsize = 10*ncols, family = 'cmr10')
                plt.yticks(fontsize = 10*ncols, family = 'cmr10')
            
            plt.tight_layout()
            # plt.subplots_adjust(top = (1-3/(10*nrows)))
            _ = gc.collect()
            
            if folder_path is not None:
                plt.savefig(f'{folder_path}/{data_name} resid decomposition result (components-{i+1}).png') 
                plt.savefig(f'{folder_path}/{data_name} resid decomposition result (components-{i+1}).pdf') 
    
    return fig1


def Standard_Scaler(trues, preds, ref_data = None):
    
    def maintain_dimension(arr, dim):
        arr = np.array(arr)
        while arr.ndim < dim:
            arr = np.expand_dims(arr, axis=-1)
        return arr
    
    trues, preds = maintain_dimension(trues, 3), maintain_dimension(preds, 3)
    trues_reshaped = trues.reshape(-1, trues.shape[-1])
    preds_reshaped = preds.reshape(-1, trues.shape[-1])
    if ref_data is None:
        scaler = StandardScaler().fit(trues_reshaped)
    else:
        ref_data = ref_data.reshape(ref_data.shape[0], -1)
        scaler = StandardScaler().fit(ref_data)
    
    trues_scaled = scaler.transform(trues_reshaped).reshape(trues.shape)
    preds_scaled = scaler.transform(preds_reshaped).reshape(preds.shape)
    return trues_scaled, preds_scaled
            

def combine_paths(path1, path2):
    # 将两个路径转换为绝对路径
    abs_path1 = os.path.abspath(path1)
    abs_path2 = os.path.abspath(path2)
    
    # 获取两个路径的根目录部分
    root1 = os.path.splitdrive(abs_path1)[0]
    root2 = os.path.splitdrive(abs_path2)[0]
    
    # 检查两个路径是否具有相同的根目录
    if root1 == root2:
        # 如果相同，返回第二个路径
        return abs_path2
    else:
        # 如果不同，将第二个路径追加到第一个路径的根目录之后
        common_root = os.path.commonpath([root1, root2])
        return os.path.join(common_root, os.path.relpath(abs_path2, common_root))        
    
    
def Time_lag(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds - hours*3600) // 60
    seconds = seconds - hours * 3600 - minutes * 60
    
    if hours > 0:
        output = f"{hours}h"
    else:
        output = ""
    if minutes > 0:
        output =  f"{output}{minutes}m"
    if seconds > 0:
        output = f"{output}{seconds}s"
        
    if output == "":
        return "0s"
    else:
        return output
    
    
class Data_Standarder:
    def __init__(self, data, target, args):
        target_values = data[target].values
        data = data.drop(target, axis = 1)
        data[target] = target_values.tolist()       # 将target调整到最后一列
        
        data_values = data.values
        self.scaler = StandardScaler().fit(data_values)
        self.data_values = self.scaler.transform(data_values)
        self.data = pd.DataFrame(self.data_values, columns=data.columns)
        self.train_data = self.data.iloc[:int(data.shape[0]*args.train_ratio)]
        self.feature_num = data.shape[1]
        self.target_scaler = StandardScaler().fit(target_values.reshape(-1,1))
        self.args = args
        
    def _inverse_target_series(self, Input_target_data):
        Input_target_data = Input_target_data.reshape(-1,self.args.pred_len) # Input_target_series.shape = (sample_num, data lag)
        
        Original_data = np.zeros_like(Input_target_data)
        for i in range(Input_target_data.shape[1]):
            Original_data[:,i] = self.target_scaler.inverse_transform(Input_target_data[:,i].reshape(-1,1)).reshape(-1)
        
        return Original_data
                

def Logger_path(folder_path, logger_name, cover = True):
    if logger_name.endswith(".txt"):
        logger_name = logger_name[:-4]
    logger_path = f"{folder_path}/{logger_name}.txt"
    
    if cover is False:
        if os.path.exists(logger_path):
            count = 0 
            files_in_folder = os.listdir(folder_path)
            for file in files_in_folder:
                file_name, file_extension = os.path.splitext(file)
                if file_extension == ".txt" and file_name.startswith(f"{logger_name}"):
                    count += 1
            logger_path = f"{folder_path}/{logger_name}_{count}.txt"
    
    logger_path = logger_path.replace("\\", "/")
    return logger_path
        
        
def Folder_path(root_folder_path, folder_name, cover = True):
    exp_folder_path = f"{root_folder_path}/{folder_name}"
    
    if (cover is False) and os.path.exists(exp_folder_path):
        count = 0 
        files_in_rootfolder = os.listdir(root_folder_path)
        for file in files_in_rootfolder:
            if os.path.isdir(os.path.join(root_folder_path, file)):
                if file.startswith(f"{folder_name}"):
                    count += 1
        exp_folder_path = f"{exp_folder_path}_{count}"
        os.makedirs(exp_folder_path)
    
    elif not os.path.exists(exp_folder_path):
        os.makedirs(exp_folder_path)
    
    exp_folder_path = exp_folder_path.replace("\\", "/")
    return exp_folder_path
    

# Feature selection tools =====================================================
# -----------------------------------------------------------------------------
'''
    The 'estimator' parameter of permutation_importance must be an object implementing 'fit',
    so we need define a Estimator.
'''
class PytorchEstimator(BaseEstimator):
    def __init__(self, model, args, device=None):
        self.model = model
        self.device = device  # 如果需要使用 GPU
        self.args = args

    def fit(self, X, y):
        # 假设模型已经训练完成，直接返回
        # X = X.reshape(-1, self.args.seq_len, X.shape[-1])
        # y = y.reshape(-1, self.args.pred_len)
        
        # self.model = ModelTrain(self.model, X, y, self.args)
        return self

    def predict(self, X):
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            inputs = torch.FloatTensor(X)
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
        return outputs.detach().cpu().numpy()

    def eval(self):
        self.model.eval()

    def train(self, mode=True):
        self.model.train(mode)

def score(estimator, X, y, args):
    estimator.eval()
    model = estimator.model     # 获取pytorch模型
    
    X_reshaped = X.reshape(-1, args.seq_len, X.shape[-1])
    # y = torch.FloatTensor(y).to(args.device)
    
    with torch.no_grad():
        inputs = torch.FloatTensor(X_reshaped)
        inputs = inputs.to(args.device)
        y_pred = model(inputs)
        y_pred = y_pred.detach().cpu().numpy()
        # y = y.detach().cpu().numpy()
        y_pred = y_pred.reshape(y.shape)
    
    torch.cuda.empty_cache()
    loss = data_loss(y, y_pred)
    return -loss.mse  # 负号是因为 permutation_importance 期望评分越高越好



def PI_eval(model, X, Y, args):
    # create estimater
    pytorch_estimator = PytorchEstimator(model, args, device=args.device)
    
    X = X.reshape(-1, X.shape[2])
    Y = Y.reshape(Y.shape[0], -1)
    
    perm = permutation_importance(pytorch_estimator, X, Y, random_state=seed, n_repeats=args.n_pi_repeats,
                                  scoring=lambda est, x, y: score(est, x, y, args))
    importances = np.array(perm.importances_mean).reshape(-1, X.shape[-1])
    Importances = np.mean(importances, axis = 0)
    
    return Importances


def Model_evaluate(data, target, args, eval_component = "vali"):
    """

    函数: 重新根据数据训练模型然后评价结果
    data: train data

    """
    args.input_size = data.shape[1]
    model = obtain_predictor(args).preditor(args)      
    criterion = nn.MSELoss()
    
    train_X, train_Y = XYData_Loader(data, args.pred_len, args.seq_len, target)
    
    if eval_component == "vali":
        eval_len = int(train_X.shape[0] * args.vali_split)
        eval_X, eval_Y = np.copy(train_X[-eval_len:]), np.copy(train_Y[-eval_len:])
    if eval_component == "train":
        eval_X, eval_Y = np.copy(train_X), np.copy(train_Y)
    if eval_component == "train-part":
        eval_len = int(train_X.shape[0] * args.vali_split)
        eval_X, eval_Y = np.copy(train_X[:-eval_len]), np.copy(train_Y[:-eval_len])
    eval_database = TensorDataset(torch.from_numpy(eval_X), torch.from_numpy(eval_Y))
    eval_dataloader = DataLoader(eval_database, batch_size=args.batch_size, shuffle=True)
    
    model = ModelTrain(model, train_X, train_Y, args)
    loss = ModelVali(model, eval_dataloader, criterion, args)
    
    torch.cuda.empty_cache()
    
    return loss


def filter_features(data, target, PI_score_data, args, r = None, Important_df = None):
    if Important_df is None:
        args.input_size = data.shape[1]
        model = obtain_predictor(args).preditor(args)               
        train_X, train_Y = XYData_Loader(data, args.pred_len, args.seq_len, target)
        
        model = ModelTrain(model, train_X, train_Y, args)   
        if PI_score_data == "vali":
            vali_len = int(train_X.shape[0] * args.vali_split)
            eval_X, eval_Y = np.copy(train_X[-vali_len:]), np.copy(train_Y[-vali_len:])
        else:
            eval_X, eval_Y = np.copy(train_X), np.copy(train_Y)
            
        importances = PI_eval(model, eval_X, eval_Y, args)
        FeatureImportance_dict = {"Feature": data.columns,
                                  "Importance": importances}
        
        if r is None:
            r = np.max([np.max(importances) * 1e-3, 0])
        
        FeatureImportance_df = pd.DataFrame(FeatureImportance_dict).set_index("Feature")
        FeatureImportance_df = FeatureImportance_df[FeatureImportance_df["Importance"] > r]
        FeatureImportance_df = FeatureImportance_df.sort_values(by = "Importance", ascending = False)
        sel_features = list(FeatureImportance_df.index)
    else:
        r = np.max([np.max(Important_df["Importance"].values)* 1e-3, 0])
        FeatureImportance_df = Important_df[Important_df["Importance"] > r]
        FeatureImportance_df = FeatureImportance_df.sort_values(by = "Importance", ascending = False)
        sel_features = list(FeatureImportance_df.index)
    if target in sel_features:
        sel_features.remove(target)
    
    return sel_features
# -----------------------------------------------------------------------------

# 检测后台的最大占用情况
class GpuMemMonitor:
    def __init__(self, gpu_id=0, interval=0.1):   # 100 ms 采一次
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.interval = interval
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.used0 = info.used // 1024**2
        self.peak = 0          # 单位 MiB
        self._stop = False
        self._thread = None

    def _poll(self):
        while not self._stop:
            used = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used // 1024**2
            self.peak = max(self.peak, used)
            time.sleep(self.interval)

    def start(self):          # 开始后台采样
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._stop = False
        self._thread.start()

    def stop(self):           # 结束并返回峰值
        self._stop = True
        self._thread.join()
        
        max_use = max([self.peak-self.used0, 0])
        return max_use

    def __del__(self):
        pynvml.nvmlShutdown()

    
class count_model_parameters:
    def __init__(self, model):
        """
        计算模型的总参数数量、可训练参数数量和非可训练参数数量。

        Args:
            model: PyTorch 模型。

        Returns:
            total_params: 总参数数量。
            trainable_params: 可训练参数数量。
            non_trainable_params: 非可训练参数数量。
        """
        self.total_params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.non_trainable_params = self.total_params - self.trainable_params
        
        estimate_memory_bits = int(self.total_params * 4 * 1.03)
        estimate_memory = ""
        GiBs = estimate_memory_bits // (1024**3)
        MBs = (estimate_memory_bits - GiBs*(1024**3)) // (1024**2)
        KBs = (estimate_memory_bits - GiBs*(1024**3) - MBs*(1024**2)) // 1024
        if GiBs > 0:
            estimate_memory = f"{estimate_memory}{GiBs}G"
        if MBs > 0:
            estimate_memory = f"{estimate_memory}{MBs}M"
        if KBs > 0:
            estimate_memory = f"{estimate_memory}{KBs}K"
        self.estimate_memory = estimate_memory
        
    def _print(self):
        print(f"Total parmas: {self.total_params}, trained params: {self.trainable_params}. Estimate memory: {self.estimate_memory}.")

def Clear_dir(path: str):
    """清空目录，但保留目录本身"""
    if os.path.exists(path):
        if not os.path.isdir(path):
            return
        for entry in os.listdir(path):
            full = os.path.join(path, entry)
            if os.path.isdir(full):
                shutil.rmtree(full)   # 递归删整个子目录
            else:
                os.remove(full)       # 删普通文件










    
    