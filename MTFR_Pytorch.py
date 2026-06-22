'''
    Combination prediction function
'''
import numpy as np
import time
import os
import torch
import pickle

from ModelUtils import TimeSeries_Exp
from tools import data_loss, Time_lag, GpuMemMonitor, count_model_parameters
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from Configs import obtain_feature_selector


def CombinationExp(model, data, train_data, args, data_name = '', 
                   upper = None, lower = None, folder_path = None):
    
    whole_start_time = time.time()
    time1 = time.time()
    class data_class():
        def __init__(self, loss, prediction_datas, true_datas, train_trues, train_preds, Models,
                     trend_trues, trend_preds, seasonal_trues, seasonal_preds, resid_trues, resid_preds):
            self.loss = loss
            self.preds = prediction_datas
            self.trues = true_datas
            self.train_trues = train_trues
            self.train_preds = train_preds
            self.Models = Models
            self.trend_preds = trend_preds
            self.trend_trues = trend_trues
            self.seasonal_trues = seasonal_trues
            self.seasonal_preds = seasonal_preds
            self.resid_trues = resid_trues
            self.resid_preds = resid_preds
    
    # logger ------------------------------------------------------------------
    if ((folder_path is not None) and (args.fored_excute != 1)):
        timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
        logger_path = f"{folder_path}/logger.txt"
        f = open(logger_path, "w")
        f.write(f"Start time: {timepoint}.\n\n")
        f.close()
        
    # 处理趋势数据 -------------------------------------------------------------
    if torch.cuda.is_available():
        # 检测显存使用
        torch.cuda.empty_cache()
        mon = GpuMemMonitor()
        mon.start()
        
    Models = {}
        
    trend = data.trend
    train_trend = train_data.trend
    if trend is None:
        trend_preds, trend_trues, train_preds, train_trues = 0, 0, 0, 0
    else:
        args.input_size = 1
        if ((folder_path is not None) and (args.fored_excute != 2)) and ('trend model.pth' in os.listdir(folder_path)):
            trend_model = model(args)
            trend_model_state = torch.load(f"{folder_path}/trend model.pth")
            trend_model.load_state_dict(trend_model_state)
            trend_result = TimeSeries_Exp(trend, train_trend, trend_model, args, CompId="trend", folder_path=folder_path)
        else:
            trend_result = TimeSeries_Exp(trend, train_trend, model, args, CompId="trend", folder_path=folder_path)
        trend_preds = trend_result.original_preds
        trend_trues = trend_result.original_trues
        Models["trend model"] = trend_result.model
        
        train_preds = np.copy(trend_result.train_preds)
        train_trues = np.copy(trend_result.train_trues)
        
        # print
        time2 = time.time()
        delta_time = time2 - time1
        time1 = time.time()
        timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        true_scaled, pred_scaled = trend_result.trues, trend_result.preds
        loss = data_loss(trend_trues.reshape(trend_trues.shape[0], -1), trend_preds.reshape(trend_preds.shape[0], -1), args.period)
        loss_scaled = data_loss(true_scaled.reshape(true_scaled.shape[0], -1), pred_scaled.reshape(pred_scaled.shape[0], -1), args.period)
            
            
        if args.verbose:
            print("---------------------------------------------------------------------------")
            print(f"** Trend item | Cost time: {delta_time:.2f}s | {timepoint}")
            print(f"    Standard: MAE:{loss_scaled.mae:7.4f}|RMSE:{loss_scaled.rmse:7.4f}|MASE:{loss_scaled.mase:7.4f}|r:{loss_scaled.corr_coef:7.4f}|r2:{loss_scaled.r2:7.4f}")
            print(f"    Original: MAE:{loss.mae:7.4f}|RMSE:{loss.rmse:7.4f}|MASE:{loss.mase:7.4f}|r:{loss.corr_coef:7.4f}|r2:{loss.r2:7.4f}")
            print("---------------------------------------------------------------------------\n")
            
        # save data
        if ((folder_path is not None) and (args.fored_excute != 1)):
            np.save(f"{folder_path}/trend preds.npy", trend_preds)
            np.save(f"{folder_path}/trend trues.npy", trend_trues)
            torch.save(trend_result.model.state_dict(), f"{folder_path}/trend model.pth")
            model_paras = count_model_parameters(trend_result.model)
            
            f = open(logger_path, "a")
            f.write(f"** Trend item | Cost time: {delta_time:.2f}s | {timepoint}\n"
                    f"    Model parameter: {model_paras.total_params}; Estimated occupied: {model_paras.estimate_memory}\n"
                    f"    Model structure: {trend_result.model}\n"
                    f"    Standard: MAE:{loss_scaled.mae:7.4f}|RMSE:{loss_scaled.rmse:7.4f}|MASE:{loss_scaled.mase:7.4f}|r:{loss_scaled.corr_coef:7.4f}|r2:{loss_scaled.r2:7.4f}\n"
                    f"    Original: MAE:{loss.mae:7.4f}|RMSE:{loss.rmse:7.4f}|MASE:{loss.mase:7.4f}|r:{loss.corr_coef:7.4f}|r2:{loss.r2:7.4f}\n")
            if torch.cuda.is_available():
                # 检测显存使用
                max_use = mon.stop()
                f.write(f'    The maximum gpu use: {max_use} MiB.\n')
            f.write('\n')
            f.close()
    
    # 处理季节性数据
    seasonals = data.seasonals
    train_seasonals = train_data.seasonals
    if seasonals is None:
        seasonal_preds, seasonal_trues = 0, 0
    else:
        for i,col in enumerate(seasonals.columns):
            if torch.cuda.is_available():
                # 检测显存使用
                torch.cuda.empty_cache()
                mon = GpuMemMonitor()
                mon.start()
                
            seasonal_data = seasonals[col].values
            train_seasonal_data = train_seasonals[col].values
            args.input_size = 1
            if ((folder_path is not None) and (args.fored_excute != 2)) and (f'{col} model.pth' in os.listdir(folder_path)):
                season_model = model(args)
                season_model_state = torch.load(f"{folder_path}/{col} model.pth")
                season_model.load_state_dict(season_model_state)
                seasonal_result = TimeSeries_Exp(seasonal_data, train_seasonal_data, season_model, args, 
                                                 CompId=col, folder_path=folder_path)
            else:
                seasonal_result = TimeSeries_Exp(seasonal_data, train_seasonal_data, model, args, 
                                                 CompId=col, folder_path=folder_path)
            seasonal_pred = seasonal_result.original_preds
            seasonal_true = seasonal_result.original_trues
            if i == 0:
                seasonal_preds = np.copy(seasonal_pred)
                seasonal_trues = np.copy(seasonal_true)
            else:
                seasonal_preds += np.copy(seasonal_pred)
                seasonal_trues += np.copy(seasonal_true)
            
            train_preds += np.copy(seasonal_result.train_preds)
            train_trues += np.copy(seasonal_result.train_trues)
            Models[f"{col} model"] = seasonal_result.model
                
            # print
            time2 = time.time()
            delta_time = time2 - time1
            time1 = time.time()
            timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            true_scaled, pred_scaled = seasonal_result.trues, seasonal_result.preds
            loss = data_loss(seasonal_true.reshape(seasonal_true.shape[0], -1), seasonal_pred.reshape(seasonal_pred.shape[0], -1), args.period)
            loss_scaled = data_loss(true_scaled.reshape(true_scaled.shape[0], -1), pred_scaled.reshape(pred_scaled.shape[0], -1), args.period)
            
            if args.verbose:
                print("---------------------------------------------------------------------------")
                print(f"** {col} item | Cost time: {delta_time:.2f}s | {timepoint}")
                print(f"    Standard: MAE:{loss_scaled.mae:7.4f}|RMSE:{loss_scaled.rmse:7.4f}|MASE:{loss_scaled.mase:7.4f}|r:{loss_scaled.corr_coef:7.4f}|r2:{loss_scaled.r2:7.4f}")
                print(f"    Original: MAE:{loss.mae:7.4f}|RMSE:{loss.rmse:7.4f}|MASE:{loss.mase:7.4f}|r:{loss.corr_coef:7.4f}|r2:{loss.r2:7.4f}")
                print("---------------------------------------------------------------------------\n")
            
            # save data
            if ((folder_path is not None) and (args.fored_excute != 1)):
                np.save(f"{folder_path}/{col} preds.npy", seasonal_pred)
                np.save(f"{folder_path}/{col} trues.npy", seasonal_true)
                torch.save(seasonal_result.model.state_dict(), f"{folder_path}/{col} model.pth")
                model_paras = count_model_parameters(seasonal_result.model)
                
                f = open(logger_path, "a")
                f.write(f"** {col} item | Cost time: {delta_time:.2f}s | {timepoint}\n"
                        f"    Model parameter: {model_paras.total_params}; Estimated occupied: {model_paras.estimate_memory}\n"
                        f"    Model structure: {seasonal_result.model}\n"
                        f"    Standard: MAE:{loss_scaled.mae:7.4f}|RMSE:{loss_scaled.rmse:7.4f}|MASE:{loss_scaled.mase:7.4f}|r:{loss_scaled.corr_coef:7.4f}|r2:{loss_scaled.r2:7.4f}\n"
                        f"    Original: MAE:{loss.mae:7.4f}|RMSE:{loss.rmse:7.4f}|MASE:{loss.mase:7.4f}|r:{loss.corr_coef:7.4f}|r2:{loss.r2:7.4f}\n")
                if torch.cuda.is_available():
                    # 检测显存使用
                    max_use = mon.stop()
                    f.write(f'    The maximum gpu use: {max_use} MiB.\n')
                f.write('\n')
                f.close()
    
    # 处理剩余项数据
    remainder = data.resid                  # dict
    train_remainder = train_data.resid      # dict
    for i,(resid_key, resid_df) in enumerate(remainder.items()):
        if torch.cuda.is_available():
            # 检测显存使用
            torch.cuda.empty_cache()
            mon = GpuMemMonitor()
            mon.start()
            
        args.score_id = resid_key
        
        if ((folder_path is not None) and (args.fored_excute != 1)):
            args.train_logger = f"{folder_path}/{resid_key} train logger.txt"
            # log_f = open(args.train_logger, "w")
            # log_f.close()

        resid_df_copy = resid_df.copy()
        train_resid_df_copy = train_remainder[resid_key].copy()
        
        selected_feature_list_path = f"{folder_path}/{resid_key} selected features.pkl"
        if os.path.exists(selected_feature_list_path):
            with open(selected_feature_list_path, 'rb') as f:
                sel_features = pickle.load(f)
        else:
            if ((folder_path is not None) and (args.fored_excute != 1)):
                args.FeatureSelection_logger = f"{folder_path}/{resid_key} feature selection logger.txt"
                log_f = open(args.FeatureSelection_logger, "w")
                log_f.close()
            
            Feature_Selection = obtain_feature_selector(args).selector
            sel_features, figs, Feature_Score_df = Feature_Selection(resid_df_copy, data.target, args)
            with open(selected_feature_list_path, 'wb') as f:
                pickle.dump(sel_features, f)
        
        sel_cols = list(set(sel_features + [data.target]))
        resid_df = resid_df_copy[sel_cols]
        train_resid_df = train_resid_df_copy[sel_cols]
        args.input_size = resid_df.shape[1]
        # resid_model = model(args)
        # resid_result = TimeSeries_Exp(resid_df, train_resid_df, model, args, target=data.target)
        if ((folder_path is not None) and (args.fored_excute != 2)) and (f'{resid_key} model.pth' in os.listdir(folder_path)):
            resid_model = model(args)
            resid_model_state = torch.load(f"{folder_path}/{resid_key} model.pth")
            resid_model.load_state_dict(resid_model_state)
            resid_result = TimeSeries_Exp(resid_df, train_resid_df, resid_model, args, target=data.target, 
                                          CompId=resid_key, folder_path=folder_path)
        else:
            resid_result = TimeSeries_Exp(resid_df, train_resid_df, model, args, target=data.target, 
                                          CompId=resid_key, folder_path=folder_path)
        resid_pred = resid_result.original_preds
        resid_true = resid_result.original_trues
        if i == 0:
            resid_preds = np.copy(resid_pred)
            resid_trues = np.copy(resid_true)
        else:
            resid_preds += np.copy(resid_pred)
            resid_trues += np.copy(resid_true)
        
        train_preds += np.copy(resid_result.train_preds)
        train_trues += np.copy(resid_result.train_trues)
        Models[f"{resid_key} model"] = resid_result.model
        
        # print
        time2 = time.time()
        delta_time = time2 - time1
        time1 = time.time()
        timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        true_scaled, pred_scaled = resid_result.trues, resid_result.preds
        loss = data_loss(resid_true, resid_pred, args.period)
        loss_scaled = data_loss(true_scaled, pred_scaled, args.period)
        if len(sel_features) > 0:
            sel_features_string = ", ".join(sel_features)
        else:
            sel_features_string = None
            
            
        if args.verbose:
            print("---------------------------------------------------------------------------")
            print(f"** {resid_key} item | Cost time: {delta_time:.2f}s | {timepoint}")
            print(f"    Selected features: {sel_features_string}")
            print(f"    Standard: MAE:{loss_scaled.mae:7.4f}|RMSE:{loss_scaled.rmse:7.4f}|MASE:{loss_scaled.mase:7.4f}|r:{loss_scaled.corr_coef:7.4f}|r2:{loss_scaled.r2:7.4f}")
            print(f"    Original: MAE:{loss.mae:7.4f}|RMSE:{loss.rmse:7.4f}|MASE:{loss.mase:7.4f}|r:{loss.corr_coef:7.4f}|r2:{loss.r2:7.4f}")
            print("---------------------------------------------------------------------------\n")
            
        # save data
        if ((folder_path is not None) and (args.fored_excute != 1)):
            np.save(f"{folder_path}/{resid_key} preds.npy", resid_pred)
            np.save(f"{folder_path}/{resid_key} trues.npy", resid_true)
            torch.save(resid_result.model.state_dict(), f"{folder_path}/{resid_key} model.pth")
            model_paras = count_model_parameters(resid_result.model)
            try:
                Feature_Score_df.to_csv(f"{folder_path}/{resid_key} Feature_Sorce.csv", index = True)
                score_path = f"{args.score_folder}/{args.data_name}/{args.score_method}/{args.seq_len}-{args.pred_len}/{args.score_id} Feature_Sorce.csv"
                if not os.path.exists(score_path):
                    score_folder = f"{args.score_folder}/{args.data_name}/{args.score_method}/{args.seq_len}-{args.pred_len}"
                    if not os.path.exists(score_folder):
                        os.makedirs(score_folder)
                    Feature_Score_df.to_csv(score_path, index = True)
                
                figs[0].savefig(f"{folder_path}/{resid_key} Importance_Sorce (selectFeature).pdf", dpi = 500, bbox_inches='tight')
                figs[1].savefig(f"{folder_path}/{resid_key} Importance_Sorce (topNumFeature).pdf", dpi = 500, bbox_inches='tight')
                figs[2].savefig(f"{folder_path}/{resid_key} Importance_Sorce (Loss change).pdf", dpi = 500, bbox_inches='tight')
                figs[3].to_csv(f"{folder_path}/{resid_key} Feature_loss.csv", index = True)   # dataframe data
            except:
                pass
            
            f = open(logger_path, "a")
            f.write(f"** {resid_key} item | Cost time: {delta_time:.2f}s | {timepoint}\n"
                    f"    Selected features: {sel_features_string}\n"
                    f"    Model parameter: {model_paras.total_params}; Estimated occupied: {model_paras.estimate_memory}\n"
                    f"    Model structure: {resid_result.model}\n"
                    f"    Standard: MAE:{loss_scaled.mae:7.4f}|RMSE:{loss_scaled.rmse:7.4f}|MASE:{loss_scaled.mase:7.4f}|r:{loss_scaled.corr_coef:7.4f}|r2:{loss_scaled.r2:7.4f}\n"
                    f"    Original: MAE:{loss.mae:7.4f}|RMSE:{loss.rmse:7.4f}|MASE:{loss.mase:7.4f}|r:{loss.corr_coef:7.4f}|r2:{loss.r2:7.4f}\n")
            if torch.cuda.is_available():
                # 检测显存使用
                max_use = mon.stop()
                f.write(f'    The maximum gpu use: {max_use} MiB.\n')
            f.write('\n')
            f.close()
        elif ((folder_path is not None) and (args.fored_excute == 1)):
            Feature_Score_df.to_csv(f"{folder_path}/{resid_key} Feature_Sorce.csv", index = True)
            score_path = f"{args.score_folder}/{args.data_name}/{args.score_method}/{args.seq_len}-{args.pred_len}/{args.score_id} Feature_Sorce.csv"
            if not os.path.exists(score_path):
                score_folder = f"{args.score_folder}/{args.data_name}/{args.score_method}/{args.seq_len}-{args.pred_len}"
                if not os.path.exists(score_folder):
                    os.makedirs(score_folder)
                Feature_Score_df.to_csv(score_path, index = True)
            
            
    
    # 结果综合
    preds = trend_preds + seasonal_preds + resid_preds
    trues = trend_trues + seasonal_trues + resid_trues
    
    if lower is not None:
        preds[preds < lower] = lower
        train_preds[train_preds < lower] = lower
    if upper is not None:
        preds[preds > upper] = upper
        train_preds[train_preds > upper] = upper
    loss = data_loss(trues.reshape(trues.shape[0], -1), preds.reshape(preds.shape[0], -1), mase_season=args.period)
    if ((folder_path is not None) and (args.fored_excute != 1)):
        np.save(f"{folder_path}/preds.npy", preds)
        np.save(f"{folder_path}/trues.npy", trues)
    
    if args.verbose:
        whole_end_time = time.time()
        whole_spend_time = (whole_end_time - whole_start_time)/3600
        timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        trues_copy, preds_copy = np.copy(trues), np.copy(preds)
        if trues_copy.ndim == 3:
            trues_copy = np.squeeze(trues_copy)
        if preds_copy.ndim == 3:
            preds_copy = np.squeeze(preds_copy)
        if trues_copy.ndim == 1:
            trues_copy = np.expand_dims(trues_copy, axis = -1)
        if preds_copy.ndim == 1:
            preds_copy = np.expand_dims(preds_copy, axis = -1)
        scaler = StandardScaler().fit(np.copy(trues_copy))
        trues_scaled = scaler.transform(np.copy(trues_copy))
        preds_scaled = scaler.transform(np.copy(preds_copy))
        loss_scaled = data_loss(trues_scaled.reshape(trues_scaled.shape[0], -1), preds_scaled.reshape(preds_scaled.shape[0], -1), args.period)
        
        print(f"** Exp.{data_name} {args.seq_len}->{args.pred_len} | Cost time: {whole_spend_time:.2f} h | {timepoint}")
        print(f"    Standard: MAE:{loss_scaled.mae:7.4f}|RMSE:{loss_scaled.rmse:7.4f}|MASE:{loss_scaled.mase:7.4f}|r:{loss_scaled.corr_coef:7.4f}|r2:{loss_scaled.r2:7.4f}")
        print(f"    Original: MAE:{loss.mae:7.4f}|RMSE:{loss.rmse:7.4f}|MASE:{loss.mase:7.4f}|r:{loss.corr_coef:7.4f}|r2:{loss.r2:7.4f}\n")
    
    return data_class(loss, preds, trues, train_trues, train_preds, Models, 
                      trend_trues, trend_preds, seasonal_trues, seasonal_preds, resid_trues, resid_preds)
        
        
        
def Direct_Prediction(model, data, train_data, target, args, data_name = '', 
                   upper = None, lower = None, folder_path = None):
    class data_class():
        def __init__(self, loss, prediction_datas, true_datas):
            self.loss = loss
            self.preds = prediction_datas
            self.trues = true_datas
            
    time1 = time.time()
    args.score_id = data_name
    
    if ((folder_path is not None) and (args.fored_excute != 1)):
        logger_path = f"{folder_path}/logger.txt"
        
        args.train_logger = f"{folder_path}/{data_name} train logger.txt"
        log_f = open(args.train_logger, "w")
        log_f.close()
        
        args.FeatureSelection_logger = f"{folder_path}/{data_name} feature selection logger.txt"
        log_f = open(args.FeatureSelection_logger, "w")
        log_f.close()
    
    if args.FSMethod:
        Feature_Selection = obtain_feature_selector(args).selector
        sel_features, figs, Feature_Score_df = Feature_Selection(data.copy(), target, args)
    else:
        sel_features = list(data.columns)
    
    sel_cols = list(set(sel_features + [target]))
    data = data[sel_cols].copy()
    train_data = train_data[sel_cols].copy()
    args.input_size = len(sel_cols)
    result = TimeSeries_Exp(data, train_data, model, args, target=target)
    preds = result.original_preds
    trues = result.original_trues
    
    if lower is not None:
        preds[preds < lower] = lower
    if upper is not None:
        preds[preds > upper] = upper
    
    # print
    time2 = time.time()
    delta_time = time2 - time1
    time1 = time.time()
    timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    trues_scaled, preds_scaled = result.trues, result.preds
    loss = data_loss(trues, preds, args.period)
    loss_scaled = data_loss(trues_scaled, preds_scaled, args.period)
    if len(sel_features) > 0:
        sel_features_string = ", ".join(sel_features)
    else:
        sel_features_string = None
        
        
    if args.verbose:
        print("---------------------------------------------------------------------------")
        print(f"** {data_name} | Not decomposition | Cost time: {Time_lag(delta_time)} | {timepoint}")
        print(f"    Selected features: {sel_features_string}")
        print(f"    Standard: MAE:{loss_scaled.mae:7.4f}|RMSE:{loss_scaled.rmse:7.4f}|MASE:{loss_scaled.mase:7.4f}|r:{loss_scaled.corr_coef:7.4f}|r2:{loss_scaled.r2:7.4f}")
        print(f"    Original: MAE:{loss.mae:7.4f}|RMSE:{loss.rmse:7.4f}|MASE:{loss.mase:7.4f}|r:{loss.corr_coef:7.4f}|r2:{loss.r2:7.4f}")
        print("---------------------------------------------------------------------------\n")
        
    # save data
    if ((folder_path is not None) and (args.fored_excute != 1)):
        np.save(f"{folder_path}/preds.npy", preds)
        np.save(f"{folder_path}/trues.npy", trues)
        torch.save(result.model.state_dict(), f"{folder_path}/{data_name} model.pth")
        try:
            # figs[0].savefig(f"{folder_path}/{resid_key} Importance_Sorce (selectFeature).png", dpi = 500, bbox_inches='tight')
            figs[0].savefig(f"{folder_path}/{data_name} Importance_Sorce (selectFeature).pdf", dpi = 500, bbox_inches='tight')
            # figs[1].savefig(f"{folder_path}/{resid_key} Importance_Sorce (topNumFeature).png", dpi = 500, bbox_inches='tight')
            figs[1].savefig(f"{folder_path}/{data_name} Importance_Sorce (topNumFeature).pdf", dpi = 500, bbox_inches='tight')
            figs[2].savefig(f"{folder_path}/{data_name} Importance_Sorce (Loss change).pdf", dpi = 500, bbox_inches='tight')
            figs[3].to_csv(f"{folder_path}/{data_name} Feature_loss.csv", index = True)   # dataframe data
            
            Feature_Score_df.to_csv(f"{folder_path}/{data_name} Feature_Sorce.csv", index = True)
            score_path = f"{args.score_folder}/{args.data_name}/{args.score_method}/{args.seq_len}-{args.pred_len}/{args.score_id} Feature_Sorce.csv"
            if not os.path.exists(score_path):
                score_folder = f"{args.score_folder}/{args.data_name}/{args.score_method}/{args.seq_len}-{args.pred_len}"
                if not os.path.exists(score_folder):
                    os.makedirs(score_folder)
                Feature_Score_df.to_csv(score_path, index = True)
        except:
            pass
        
        f = open(logger_path, "a")
        f.write(f"** {data_name} | Not decomposition | Cost time: {Time_lag(delta_time)} | {timepoint}\n"
                f"    Selected features: {sel_features_string}\n"
                f"    Standard: MAE:{loss_scaled.mae:7.4f}|RMSE:{loss_scaled.rmse:7.4f}|MASE:{loss_scaled.mase:7.4f}|r:{loss_scaled.corr_coef:7.4f}|r2:{loss_scaled.r2:7.4f}\n"
                f"    Original: MAE:{loss.mae:7.4f}|RMSE:{loss.rmse:7.4f}|MASE:{loss.mase:7.4f}|r:{loss.corr_coef:7.4f}|r2:{loss.r2:7.4f}\n\n")
        f.close()
    elif ((folder_path is not None) and (args.fored_excute == 1)):
        try:
            Feature_Score_df.to_csv(f"{folder_path}/{data_name} Feature_Sorce.csv", index = True)
            score_path = f"{args.score_folder}/{args.data_name}/{args.score_method}/{args.seq_len}-{args.pred_len}/{args.score_id} Feature_Sorce.csv"
            if not os.path.exists(score_path):
                score_folder = f"{args.score_folder}/{args.data_name}/{args.score_method}/{args.seq_len}-{args.pred_len}"
                if not os.path.exists(score_folder):
                    os.makedirs(score_folder)
                Feature_Score_df.to_csv(score_path, index = True)
        except:
            pass
    
    return data_class(loss, preds, trues)        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    