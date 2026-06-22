import pandas as pd
import numpy as np
import os
import time
import shap
import torch
import matplotlib.pyplot as plt

from tools import plot_FeatureImportance, XYData_Loader
from sklearn.preprocessing import StandardScaler
from ModelUtils import ModelTrain
from tools import Model_evaluate, Time_lag
from datetime import datetime
from Models import obtain_predictor
   
    
def SHAP_eval(model, data, args):
    '''
        SHAP的部分操作可能不支持GPU操作, 所以要全部放在cpu上
    '''
    if args.verbose == 0:
        time1 = time.time()
        timepoint = datetime.now().strftime('%Y-%m-%d %H:%M')
        print(f"\tStart to evaluate the features by deepSHAP... [{timepoint}]")
    vali_len = int(data.shape[0] * args.vali_split)
    Input_data = torch.tensor(data[:-vali_len], dtype=torch.float32).to(args.device)
    explain_data = torch.tensor(data[-vali_len:], dtype=torch.float32).to(args.device)
    # Input_data = torch.tensor(data, dtype=torch.float32).to(args.device)
    # explain_data = torch.tensor(data, dtype=torch.float32).to(args.device)
    
    model.to(args.device)
    # model.eval()
    model.train()
    explainer = shap.DeepExplainer(model, Input_data)
    try:
        model.train()
        shap_values = explainer.shap_values(explain_data)  
    except:
        model.train()
        shap_values = explainer.shap_values(explain_data, check_additivity=False)  # 禁用加和校验
    
    Importances = np.mean(np.abs(np.array(shap_values)), axis = (0,1,2)).tolist()
    torch.cuda.empty_cache()
    
    if args.verbose == 0:
        time2 = time.time()
        timepoint = datetime.now().strftime('%Y-%m-%d %H:%M')
        print(f"\tFinish evaluating ... [{timepoint}|Time:{Time_lag(time2-time1)}]")
    return Importances  


def ThreePartIdxs(whole_len, extract_len):
    if (extract_len is None) or (extract_len is False):
        extract_len = whole_len
    if whole_len > extract_len:
        comp1_len = int(extract_len/3)
        comp2_len = comp1_len
        comp3_len = extract_len - comp1_len - comp2_len
        
        # component 2
        start_idx2 = np.random.randint(comp1_len, whole_len-comp3_len-comp2_len)
        comp2_idx = np.arange(start_idx2, start_idx2 + comp2_len).astype(int).tolist()
        
        # component 1
        start_idx1 = np.random.randint(0, start_idx2-comp1_len)
        comp1_idx = np.arange(start_idx1, start_idx1 + comp1_len).astype(int).tolist()
        
        # component 3
        start_idx3 = np.random.randint(start_idx2 + comp2_len, whole_len - comp3_len)
        comp3_idx = np.arange(start_idx3, start_idx3 + comp3_len).astype(int).tolist()
        
        idx = comp1_idx + comp2_idx + comp3_idx
    
    else:
        Len = whole_len
        idx = np.arange(Len).astype(int).tolist()
        
    return idx



def Feature_Selection(data, target, args):
    '''

    Parameters
    ----------
    data : train_data.
    

    '''
    
    if args.FeatureSelection_logger:
        whole_logger_time1 = time.time()
    
    # extract data
    targetSerise = data[target].values.tolist()
    data.drop(target, axis = 1, inplace = True)
    data[target] = targetSerise
    
    scaler = StandardScaler().fit(data.values)
    data_scaled = scaler.transform(data.values)
    Train = pd.DataFrame(data_scaled, columns=data.columns)
    
    train_X, train_Y = XYData_Loader(Train, args.pred_len, args.seq_len, target)
    
    # evaluation data set
    if "train" in args.score_method:
        eval_X, eval_Y = np.copy(train_X), np.copy(train_Y)
    if "vali" in args.score_method:
        vali_len = int(train_X.shape[0] * args.vali_split)
        eval_X, eval_Y = np.copy(train_X[-vali_len:]), np.copy(train_Y[-vali_len:])
    
    # model train ------------------------------------------------------------
    args.input_size = data.shape[1]
    model = obtain_predictor(args).preditor(args)                                  
    model = ModelTrain(model, train_X, train_Y, args)    
    
    if args.FeatureSelection_logger:
        log_f = open(args.FeatureSelection_logger, "a")
        timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
        log_f.write(f"Start to select features ... | {timepoint}\n")
        log_f.write("\tFinshing the initializing training.\n")
        log_f.close()
        log_time1 = time.time()
    
    # important score --------------------------------------------------------
    timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    score_path = f"{args.score_folder}/{args.data_name}/{args.score_method}/{args.seq_len}-{args.pred_len}/{args.score_id} Feature_Sorce.csv"
    if os.path.exists(score_path):
        score_folder = f"{args.score_folder}/{args.data_name}/{args.score_method}/{args.seq_len}-{args.pred_len}"
        Importance_df = pd.read_csv(score_path, index_col="Feature")
    else:
        score_folder = f"{args.score_folder}/{args.data_name}/{args.score_method}/{args.seq_len}-{args.pred_len}"
        if not os.path.exists(score_folder):
            os.makedirs(score_folder)
        Input_idx = ThreePartIdxs(train_X.shape[0], args.extract_n)
        importances = SHAP_eval(model, train_X[Input_idx], args)
        # importances = SHAP_eval(model, train_X, args)
        FeatureImportance_dict = {"Feature": data.columns,
                                  "Importance": importances}
        Importance_df = pd.DataFrame(FeatureImportance_dict).set_index("Feature")
        Importance_df.to_csv(score_path, index = True)
    if target in list(Importance_df.index):
        Importance_df.drop(target, axis = 0, inplace = True)
    
    Importance_df = Importance_df.sort_values(by = "Importance", ascending = False)
    features_list = list(Importance_df.index)
    
    if args.FeatureSelection_logger:
        log_f = open(args.FeatureSelection_logger, "a")
        timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
        log_time2 = time.time()
        log_f.write(f"\tFinshing feature evaluation | Time:{Time_lag(log_time2-log_time1)} | {timepoint} \n")
        log_f.close()
        log_id = 0
        log_time1 = time.time()
        
    # selecting features ------------------------------------------------------
    # single-series
    best_loss = Model_evaluate(Train[[target]], target, args)
    best_loss0 = best_loss
    best_features = []
    Losses = [best_loss]
    Conditions = ["single"]
    
    if args.FeatureSelection_logger:
        log_id += 1
        log_f = open(args.FeatureSelection_logger, "a")
        timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
        log_f.write(f"No.{log_id:2d} Finshing evaluating the single feature:\n")
        log_time2 = time.time()
        log_f.write(f"\tLoss: {best_loss:.4f} | Time:{Time_lag(log_time2-log_time1)} | {timepoint} \n")
        log_time1 = time.time()
        log_f.close()
    
    # class selection
    feature_id = 0 
    patience_count = 0
    do = True
    while do:
        sel_features = best_features[:] + [features_list[feature_id]]
        sel_cols = list(set(sel_features + [target]))
        loss = Model_evaluate(Train[sel_cols], target, args)
        
        if loss < best_loss:
            best_loss0 = best_loss
            best_loss = loss
            best_features = sel_features[:]         # deep copy
            patience_count = 0
        else:
            patience_count += 1
            
        feature_id += 1
        Losses.append(loss)
        Conditions.append(feature_id)
        
        if feature_id >= len(features_list) or patience_count >= args.patience:
            do = False
        
        if args.FeatureSelection_logger:
            log_id += 1
            log_f = open(args.FeatureSelection_logger, "a")
            timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
            log_f.write(f"No.{log_id:2d} Finshing evaluating the feature:\n")
            if len(best_features) > 0:
                best_features_string = ", ".join(best_features)
                log_f.write(f"\tBest features:{best_features_string}\n")
            
            log_time2 = time.time()
            log_f.write(f"\tLoss: {loss:.4f} | Time:{Time_lag(log_time2-log_time1)} | {timepoint} \n")
            log_time1 = time.time()
            
            if do == False:
                log_f.write("Stop ... \n")
            log_f.close()

        
    Featrues_sorted_df = Importance_df.loc[best_features].copy().sort_values(by = "Importance", ascending = False)
    Features_sorted = list(Featrues_sorted_df.index)
    
    Losses_df = pd.DataFrame({"Condition": Conditions, "Loss": Losses})
    # Losses_df.to_csv(f"{score_folder}/{args.FSMethod} - {args.score_id} Feature_loss.csv")
    
    # 画图 --------------------------------------------------------------------
    selectNum = np.min([args.select_features_n, data.shape[1]-1, len(features_list)])
    fig_selectFeature = plot_FeatureImportance(Featrues_sorted_df, topId=1)
    
    Importance_df2 = Importance_df.copy()
    Importance_df2["Class"] = np.zeros(Importance_df2.shape[0]).tolist()
    for f in Features_sorted:
        Importance_df2.loc[f,"Class"] = 1
    fig_topNumFeature = plot_FeatureImportance(Importance_df2.iloc[:selectNum], topId=1)
    
    # 变化图
    LossChange_fig, ax = plt.subplots(figsize = (np.min([len(Losses),20]).astype(int), 5))
    ax.plot(Losses, marker = ".", c = "black")
    ax.set_title(target, family = "cmr10")
    ax.set_xticks(range(len(Losses)))
    ax.set_xticklabels(Conditions, rotation = 45)
    
    figs = (fig_selectFeature, fig_topNumFeature, LossChange_fig, Losses_df)
    
    if args.verbose:
        timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"\tFinish Feature selection... [{timepoint}]")
    
    if args.FeatureSelection_logger:
        log_f = open(args.FeatureSelection_logger, "a")
        timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
        log_time2 = time.time()
        log_f.write(f"\nFinish Feature selection... | Time:{Time_lag(log_time2 - whole_logger_time1)} | {timepoint}\n")
        log_f.close()
    
    return Features_sorted, figs, Importance_df


    
    
    