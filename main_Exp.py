'''
    指定背景的实验
'''
import sys
import numpy as np
import pandas as pd
import random
import time
import os
import traceback
import torch
import warnings 
import matplotlib.pyplot as plt

from tools import data_loss, Folder_path, Logger_path, Data_Standarder, Time_lag,\
                    Standard_Scaler, Clear_dir
from datetime import datetime
from MTFR_Pytorch import CombinationExp
from Configs import Paras, obtain_decomposer
from Models import obtain_predictor
from copy import deepcopy

current_time = datetime.now().strftime('%Y%m%d %H%M')
today = datetime.today().strftime("%Y%m%d")
warnings.filterwarnings("ignore")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed) # 0
random.seed(seed)    # 666


if __name__ == "__main__":
    whole_start_time = time.time()
    
    # settings ===========================================================
    fored_excute = 0                # Option: 0 (default),1,2,3
                                    # 0: Run the current version. The result directly outputs if exists; 
                                    # 1: Run without saving the results; 
                                    # 2: Covering all previous results; 
                                    # 3: Run current version and save results again.
    data_name = "SILO"              # dataset name.
    file_path = r"./datasets/SILO_Cairns.csv"   # the file path of dataset.
    target = 'daily_rain'           # the target variable.
    periods = [365]                 # the periodic values. Single-seasonal data: int/list; Multi-seasonal data: list.
    resid_num = 5                   # custom. the N_D value. type: int.
    pred_len = 1                    # custom. the prediction horizon. type: int.
    seq_len = 7                     # custom. the input length. type: int.
    
    # decomposer. 
    decompMethods = [
            "SSTL_UGDFT",       
            "SSTL_MWT"
        ]
    wavelet = "db10"                    # custom. Any wavelet basis supported in the PyWavelets package.
    
    # predictor
    PredictModel = "GRU"                
    GRU_units = seq_len*2               # custom. type: int.
    Linear_units = pred_len*2           # custom. type: int.
    
    # feature selection
    FSMethod = "PI"                     # option: PI (default), DeepSHA, ALL
    PFI_repeat = 10                     # custom. type: int.
    FS_Eval_datatype = "train"          # option: train (default), vali
    score_folder = r"./result/Score"    # custom. The folder where the score information is saved.
    
    # saved folders 
    folder_path = r"./result"           # custom. The folder where the results are saved.
    
    
    # Exp ===================================================================
    ExpId = 0
    for decompMethod in decompMethods:
        data_decomposer = obtain_decomposer(decompMethod, wavelet)
        decomp_method = data_decomposer.decomp_method
        current_decomp_method = data_decomposer.decompMethod_name
    
        model_time1 = time.time()
        
        # Redirect console output to a log file -----------------------------------
        original_stdout = sys.stdout
        original_stderr = sys.stderr
       
        exp_folder_path = Folder_path(os.path.abspath(folder_path), "MTFR_Exp", cover=True)
        EachExp_folder = r"./result/ALLSingleExp"
        if not os.path.exists(EachExp_folder):
            os.makedirs(EachExp_folder)
        
        # record ------------------------------------------------------------------
        logger_path = Logger_path(exp_folder_path, f"SingleExp-{FSMethod}_{current_decomp_method}_{PredictModel}-{today}", cover=True)
        
        f = open(logger_path, "w")
        f.write("=============================================================================\n")
        f.write(
                f"Forecasting method: {PredictModel} "
                f"| decomposition method: {current_decomp_method} "
                f"| score method: {FSMethod}-{FS_Eval_datatype}-{current_decomp_method}-{PredictModel} "
                f"| feature selection method: {FSMethod} "
                )
        f.write("\n")
        f.write("\n")
        f.close()
        # ------------------------------------------------------------------------
        
        
        
        cols = ['data', 'model', 'task', 'RMSE', 'MAE', 'SDE', 'TIC', 'R2',
                'R', 'EVS', 'IA', 'MAPE', 'SMAPE', 'MASE']
        Loss = pd.DataFrame(columns=cols)
        
        
            
        df = pd.read_csv(file_path, index_col='date').select_dtypes(np.number)
        
        if "PI" in FSMethod:
            FSMethod_str = f"{FSMethod}({PFI_repeat})"
        else:
            FSMethod_str = FSMethod
      
        timepoint = datetime.now().strftime('%Y-%m-%d %H:%M')
        print("------------------------ Task ------------------------")
        print(f"DataName:{data_name}\n"
              f"Model:{FSMethod_str}-{current_decomp_method}-{PredictModel}\n"
              f"Time:{timepoint}\n"
              f"Target variable: {target}\n"
              f"Resid_decomposition_number:{resid_num}")
        print("------------------------------------------------------")
        
        # record --------------------------------------------------------------
        f = open(logger_path, "a")
        f.write("------------------------ Task ------------------------\n")
        f.write(f"DataName:{data_name}\n"
                f"Model:{FSMethod_str}-{current_decomp_method}-{PredictModel}\n"
                f"Time:{timepoint}\n"
                f"Target variable: {target}\n"
                f"Resid_decomposition_number:{resid_num}\n")
        f.write("------------------------------------------------------")
        f.write("\n")
        f.write("\n")
        f.close()
        # ---------------------------------------------------------------------
        
        ExpId += 1
        decompId = 0
        time1 = time.time()
                
        torch.manual_seed(seed)
        np.random.seed(seed) # 0
        random.seed(seed)    # 666
        
        if np.ndim(resid_num) == 0:
            resid_decompNum = resid_num
        elif np.ndim(resid_num) == 1:
            resid_decompNum = resid_num[ExpId-1]
        Model = f"{FSMethod_str}-{current_decomp_method}({resid_decompNum})-{PredictModel}"
        

        single_folder_path = f"{exp_folder_path}/{data_name}/{Model}/{seq_len}-{pred_len}"
        if os.path.exists(single_folder_path) and ("original_trues.npy" in os.listdir(single_folder_path)) and (fored_excute == 0):
            args = Paras()
            args.period = np.max(periods).astype(int)
            
            trues = np.load(f"{single_folder_path}/original_trues.npy")
            preds = np.load(f"{single_folder_path}/original_preds.npy")
            trues_scaled = np.load(f"{single_folder_path}/trues.npy")
            preds_scaled = np.load(f"{single_folder_path}/preds.npy")
            
            loss = data_loss(trues, preds, mase_season=args.period)
            loss_scaled = data_loss(trues_scaled, preds_scaled, args.period)
            
            # save the loss information ----------------------------
            loss_inf = [f'{data_name}',f'{Model}',f'{seq_len}->{pred_len}', 
                        loss.rmse, loss.mae, loss.error_std, loss.tic,
                        loss.r2, loss.corr_coef, loss.evs, loss.ia, loss.mape,
                        loss.smape, loss.mase]
            loss_df = pd.DataFrame([loss_inf], columns=cols)
            Loss = pd.concat([Loss, loss_df], ignore_index=True)
            Loss.to_csv(f'{exp_folder_path}/{Model} Loss information.csv', index = False)
            
            time2 = time.time()
            timepoint = datetime.now().strftime('%Y-%m-%d %H:%M')
            print(f'Exp.{ExpId} - {data_name}|MTFR:{Model}|{seq_len}->{pred_len}|time:{Time_lag(time2-time1)}|{timepoint} >>')
            print(f"    Standard: MAE:{loss_scaled.mae:7.4f}|RMSE:{loss_scaled.rmse:7.4f}|MASE:{loss_scaled.mase:7.4f}|r:{loss_scaled.corr_coef:7.4f}|r2:{loss_scaled.r2:7.4f}")
            print(f"    Original: MAE:{loss.mae:7.4f}|RMSE:{loss.rmse:7.4f}|MASE:{loss.mase:7.4f}|r:{loss.corr_coef:7.4f}|r2:{loss.r2:7.4f}\n")
            
            # record the result ---------------------------------------
            f = open(logger_path, "a")
            f.write(f'Exp.{ExpId} - {data_name}|MTFR:{Model}|{seq_len}->{pred_len}|time:{Time_lag(time2-time1)}|{timepoint} >>\n')
            f.write(f"    Standard: MAE:{loss_scaled.mae:7.4f}|RMSE:{loss_scaled.rmse:7.4f}|MASE:{loss_scaled.mase:7.4f}|r:{loss_scaled.corr_coef:7.4f}|r2:{loss_scaled.r2:7.4f}\n")
            f.write(f"    Original: MAE:{loss.mae:7.4f}|RMSE:{loss.rmse:7.4f}|MASE:{loss.mase:7.4f}|r:{loss.corr_coef:7.4f}|r2:{loss.r2:7.4f}\n")
            f.write("\n")
            f.close()
            # ---------------------------------------------------------
            
            time1 = time.time()
            
            continue
            
        if not os.path.exists(single_folder_path):
            os.makedirs(single_folder_path)
        
        try:
            SingleExp_folder_path = f"{EachExp_folder}/{data_name}/{Model}/{seq_len}-{pred_len}"
            score_method = f"{FSMethod_str}-{FS_Eval_datatype}-{current_decomp_method}({resid_decompNum})-{PredictModel}"
            single_score_folder_path = f"{score_folder}/{data_name}/{score_method}/{seq_len}-{pred_len}"
            if not os.path.exists(SingleExp_folder_path):
                os.makedirs(SingleExp_folder_path)
            elif fored_excute == 2:
                Clear_dir(SingleExp_folder_path)
                Clear_dir(single_folder_path)
                Clear_dir(single_score_folder_path)
            
            args = Paras()
            # basic
            args.data_name = data_name
            args.pred_len = pred_len
            args.seq_len = seq_len
            args.period = np.max(periods).astype(int)
            # predictor
            args.model = PredictModel
            args.GRU_units = GRU_units
            args.Linear_units = Linear_units
            # feature selection
            args.FSMethod = FSMethod
            args.score_folder = score_folder
            args.eval_dataset_type = FS_Eval_datatype  
            args.resid_decompNum = resid_decompNum
            args.score_method = score_method
            args.n_pi_repeats = PFI_repeat
            
            data_df = df.copy()
            train_df = df.iloc[:int(df.shape[0]*args.train_ratio)].copy()
            # decomposition
            if decompId == 0:
                decomp = deepcopy(decomp_method(data_df, periods, target, resid_decompNum))
                train_decomp = deepcopy(decomp_method(train_df, periods, target, resid_decompNum))
                decompId = 1        # decompId == 1 illustrate the data has been decomposed (with same resid_num)
            elif np.ndim(resid_num) == 1:
                decomp = deepcopy(decomp_method(data_df.copy(), periods, target, resid_num[ExpId-1]))
                train_decomp = deepcopy(decomp_method(train_df.copy(), periods, target, resid_num[ExpId-1]))
                decompId = 2        # decompId == 2 illustrate the data has been decomposed (with different resid_num)
            
            # Pytorch ---------------------------------------------------
            predictor = obtain_predictor(args).preditor
            ExpResults = CombinationExp(predictor, decomp, train_decomp, args, data_name=data_name,  
                                        folder_path=SingleExp_folder_path)
            
            
            # show the evaluation result
            trues, preds = ExpResults.trues, ExpResults.preds
            loss = data_loss(trues, preds, mase_season=args.period)
            trues_scaled, preds_scaled = Standard_Scaler(np.copy(trues), np.copy(preds), df[target].values)
            loss_scaled = data_loss(trues_scaled, preds_scaled, args.period)
            
            
            # save the loss information ----------------------------
            loss_inf = [f'{data_name}',f'{Model}',f'{seq_len}->{pred_len}', 
                        loss.rmse, loss.mae, loss.error_std, loss.tic,
                        loss.r2, loss.corr_coef, loss.evs, loss.ia, loss.mape,
                        loss.smape, loss.mase]
            loss_df = pd.DataFrame([loss_inf], columns=cols)
            Loss = pd.concat([Loss, loss_df], ignore_index=True)
            Loss.to_csv(f'{exp_folder_path}/{Model} Loss information.csv', index = False)
            
            # save ------------------------------------------------
            np.save(f"{single_folder_path}/trues.npy", trues_scaled)
            np.save(f"{single_folder_path}/preds.npy", preds_scaled)
            np.save(f"{single_folder_path}/original_trues.npy", trues)
            np.save(f"{single_folder_path}/original_preds.npy", preds)
            
            time2 = time.time()
            timepoint = datetime.now().strftime('%Y-%m-%d %H:%M')
            print(f'Exp.{ExpId} - {data_name}|MTFR:{Model}|{seq_len}->{pred_len}|time:{Time_lag(time2-time1)}|{timepoint} >>')
            print(f"Train/Vail dataset size: {train_df.shape[0]}; Test dataset size: {data_df.shape[0]-train_df.shape[0]}")
            print(f"    Standard: MAE:{loss_scaled.mae:7.4f}|RMSE:{loss_scaled.rmse:7.4f}|MASE:{loss_scaled.mase:7.4f}|r:{loss_scaled.corr_coef:7.4f}|r2:{loss_scaled.r2:7.4f}")
            print(f"    Original: MAE:{loss.mae:7.4f}|RMSE:{loss.rmse:7.4f}|MASE:{loss.mase:7.4f}|r:{loss.corr_coef:7.4f}|r2:{loss.r2:7.4f}\n")
            
            
            # record the result ---------------------------------------
            f = open(logger_path, "a")
            f.write(f'Exp.{ExpId} - {data_name}|MTFR:{Model}|{seq_len}->{pred_len}|time:{Time_lag(time2-time1)}|{timepoint} >>\n')
            f.write(f"    Standard: MAE:{loss_scaled.mae:7.4f}|RMSE:{loss_scaled.rmse:7.4f}|MASE:{loss_scaled.mase:7.4f}|r:{loss_scaled.corr_coef:7.4f}|r2:{loss_scaled.r2:7.4f}\n")
            f.write(f"    Original: MAE:{loss.mae:7.4f}|RMSE:{loss.rmse:7.4f}|MASE:{loss.mase:7.4f}|r:{loss.corr_coef:7.4f}|r2:{loss.r2:7.4f}\n")
            f.write("\n")
            f.close()
            # ---------------------------------------------------------
            del decomp, train_decomp
            time1 = time.time()
            
            
        except Exception as e:
            time2 = time.time()
            timepoint = datetime.now().strftime('%Y-%m-%d %H:%M')
            print(f'Exp.{ExpId} - {data_name}|MTFR:{Model}|{seq_len}->{pred_len}|time:{Time_lag(time2-time1)}|{timepoint} >>')
            print(f'An error ocurres: {e}.\n')
            traceback.print_exc()
            time1 = time.time()
                
        
        model_time2 = time.time()
        print(f"The experiment for the current model spends {Time_lag(model_time2-model_time1)}.")
    
    whole_end_time = time.time() 
    print(f'The MTFR experiment spend {Time_lag(whole_end_time - whole_start_time)}.')
    print(f"The results are save in {exp_folder_path}.")
    print(f"The results are recorded in {logger_path}.")
    sys.stdout = original_stdout
    sys.stderr = original_stderr