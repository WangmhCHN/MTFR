import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import time
import traceback
import os

from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

   
        
# EarlyStop 
class EarlyStopping:
    def __init__(self, patience = 3, delta = 0, verbose = False, logger_path = None):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.count = 0
        self.best_score = None
        self.early_stop = False
        self.logger_path = logger_path
        # self.best_model = None
    
    def __call__(self, vali_loss, model):
        if self.best_score is None:
            self.best_score = vali_loss
            self.best_state_dict = model.state_dict()
            if self.verbose:
                print(f"Vaildation loss: Inf --> {vali_loss: .6f}")
            if self.logger_path:
                log_f = open(self.logger_path, "a")
                log_f.write(f"Vaildation loss: Inf --> {vali_loss: .6f}\n")
                log_f.close()
        elif vali_loss < self.best_score + self.delta:
            if self.verbose:
                print(f"Vaildation loss: {self.best_score:.6f} --> {vali_loss: .6f}")
            if self.logger_path:
                log_f = open(self.logger_path, "a")
                log_f.write(f"Vaildation loss: {self.best_score:.6f} --> {vali_loss: .6f}\n")
                log_f.close()
            self.best_score = vali_loss
            self.best_state_dict = model.state_dict()
            self.count = 0
        else:
            self.count += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.count} out of {self.patience}')
            if self.logger_path:
                log_f = open(self.logger_path, "a")
                log_f.write(f"EarlyStopping counter: {self.count} out of {self.patience}\n")
                log_f.close()
            if self.count >= self.patience:
                self.early_stop = True
    
    def get_best_state_dict(self):
        '''返回最佳的state_dict'''
        return self.best_state_dict

def adjust_learning_rate(optimizer, epoch, execution_time, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj is not False and execution_time < args.lradj_time:
        if args.lradj == 'type1':
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif args.lradj == 'type2':
            lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
        elif args.lradj == "cosine":
            lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.epochs * math.pi))}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if args.verbose:
                print('Updating learning rate to {}'.format(lr))
            if args.train_logger:
                log_f = open(args.train_logger, "a")
                log_f.write('Updating learning rate to {}\n'.format(lr))
                log_f.close()


# Model validation
def ModelVali(model, vali_dataloader, criterion, args):
    model.eval()
    vali_loss = []

    with torch.no_grad():
        for i, (vali_X, vali_Y) in enumerate(vali_dataloader):
            vali_X = vali_X.float().to(args.device)
            vali_Y = vali_Y.float().to(args.device)
            
            output = model(vali_X)
            output = output.reshape(vali_Y.shape)
            loss = criterion(vali_Y.detach(), output.detach())
            vali_loss.append(loss.item())

    vali_loss = np.mean(vali_loss)
    model.train()
    return vali_loss
    

# Model train 
def ModelTrain(model, X, Y, args):
    
    # data process
    # train_lag = int(X.shape[0] * (1-args.vali_split))
    # train_X, vali_X = np.array(X[:train_lag]), np.array(X[train_lag-args.seq_len:])
    # train_Y, vali_Y = np.array(Y[:train_lag]), np.array(Y[train_lag-args.seq_len:])
    # TrainDataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_Y))
    # trainloader = DataLoader(TrainDataset, batch_size=args.batch_size, shuffle=True)
    # ValiDataset = TensorDataset(torch.from_numpy(vali_X), torch.from_numpy(vali_Y))
    # valiloader = DataLoader(ValiDataset, batch_size=args.batch_size)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    train_size = int((1 - args.vali_split) * len(dataset))
    vali_size = len(dataset) - train_size
    train_dataset, vali_dataset = torch.utils.data.random_split(dataset, [train_size, vali_size])
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valiloader = DataLoader(vali_dataset, batch_size=args.batch_size)
    
    # loss function ,optimizer and device
    device = args.device
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    TrainSteps = len(trainloader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=args.verbose, 
                                   logger_path=args.train_logger)
    
    
    if args.train_logger:
        timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
        log_f = open(args.train_logger, "a")
        log_f.write("----------------------------------------------------------\n")
        log_f.write(f"Start to train... | {timepoint}")
        log_f.close()
    
    epoch_time1 = time.time()
    for epoch in range(args.epochs):
        
        model.train()
        train_loss = []
        batch_steps = 0
        batch_time1 = time.time()
        
        for i,(batch_X, batch_Y) in enumerate(trainloader):
            # if args.verbose:
            #     print(f"Iter.{i+1} batch_x.shape={batch_X.shape}, batch_y.shape={batch_Y.shape}")
            
            batch_steps += 1
            
            batch_X = batch_X.float().to(device)
            batch_Y = batch_Y.float().to(device)
            batch_output = model(batch_X)
            batch_output = batch_output.reshape(batch_Y.shape)
            
            loss = criterion(batch_output, batch_Y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()       # 清除计算梯度
            loss.backward()             # 方向传播
            optimizer.step()            # 更新模型权重
           
            
            if args.verbose:
                if (i+1) % (TrainSteps//5) == 0:
                    batch_time2 = time.time()
                    batch_time = (batch_time2 - batch_time1)/batch_steps
                    batch_steps = 0
                    timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
                    print(f"\titers: {i+1}/{TrainSteps},epoch: {epoch+1} | loss: {loss.item(): .6f} | cost time:{batch_time:.2f}s/iter | {timepoint}")
            
            if args.train_logger:
                if (i+1) % (TrainSteps//5) == 0:
                    batch_time2 = time.time()
                    batch_time = (batch_time2 - batch_time1)/batch_steps
                    batch_steps = 0
                    timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
                    log_f = open(args.train_logger, "a")
                    log_f.write(f"\titers: {i+1}/{TrainSteps},epoch: {epoch+1} | loss: {loss.item(): .6f} | cost time:{batch_time:.2f}s/iter | {timepoint}\n")
                    log_f.close()
        
        train_loss = np.mean(train_loss)
        torch.cuda.empty_cache()
        epoch_time2 = time.time()
        epoch_time = epoch_time2 - epoch_time1
        epoch_time1 = time.time()
        
        
        if args.lradj is not False:
            try:    
                adjust_learning_rate(optimizer, epoch, epoch_time, args)    
            except Exception as e:
                if args.verbose:
                    print(f"Can not early stop or adjust learning rate! A error occurs: {e}.")
                    traceback.print_exc()
                    
        if args.isEarlyStopping:
            vali_loss = ModelVali(model, valiloader, criterion, args)
            early_stopping(vali_loss, model)
            if args.verbose:
                timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
                print(f"Epoch:{epoch+1} | Train loss: {train_loss:.6f} | Vali loss: {vali_loss: .6f}")
                print(f"Spend time:{epoch_time: .2f}s | {timepoint}")
            if args.train_logger:
                timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
                log_f = open(args.train_logger, "a")
                log_f.write(f"Epoch:{epoch+1} | Train loss: {train_loss:.6f} | Vali loss: {vali_loss: .6f}\n")
                log_f.write(f"Spend time:{epoch_time: .2f}s | {timepoint}\n")
                log_f.close()
            if early_stopping.early_stop:
                break
        else:
            if args.verbose:
                timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
                print(f"Epoch:{epoch+1} | Train loss: {train_loss:.6f} | Spend time:{epoch_time: .2f}s | {timepoint}")
            if args.train_logger:
                timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
                log_f = open(args.train_logger, "a")
                log_f.write(f"Epoch:{epoch+1} | Train loss: {train_loss:.6f} | Spend time:{epoch_time: .2f}s | {timepoint}\n")
                log_f.close()
        
    if args.isEarlyStopping:
        best_state_dict = early_stopping.get_best_state_dict()
        model.load_state_dict(best_state_dict)
    
    return model



# model test
def Modelpredict(model, test_X, test_Y, scaler, args, CompId = None, folder_path = None):
    class predictData():
        def __init__(self, preds, trues, original_trues, original_preds, model):
            self.preds = preds
            self.trues = trues
            self.original_preds = original_preds
            self.original_trues = original_trues
            self.model = model
    
    time1 = time.time()
    model = model.to(args.device)
    TestDataset = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_Y))
    test_loader = DataLoader(TestDataset, batch_size=args.batch_size)
    preds = []
    trues = []
    batchX= []
    test_steps = len(test_loader)
    test_loss = []
    criterion = nn.MSELoss()
    
    model.eval()
    batch_time1 = time.time()
    batch_step = 0
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_step += 1
            
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            
            batch_output = model(batch_x)
            batch_output = batch_output.reshape(batch_y.shape)
            
            pred = batch_output.detach().cpu()
            true = batch_y.detach().cpu()
            
            pred = pred.reshape(true.shape)
              
            if args.verbose:
                loss = nn.MSELoss()(pred, true)
                test_loss.append(loss.item())
                if (i+1)%(test_steps//3) == 0:
                    batch_time2 = time.time()
                    batch_time = (batch_time2 - batch_time1)/batch_step
                    batch_step = 0
                    batch_time1 = time.time()
                    timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
                    print(f"\titers: {i+1}/{test_steps} | test loss: {loss.item():.6f} | cost time:{batch_time:.2f}s/iter | {timepoint}")
            
            preds.append(pred.numpy())
            trues.append(true.numpy())
            batchX.append(batch_x.cpu().numpy())
    
    
    if len(preds)>0:
        preds=np.concatenate(preds, axis=0)
        trues=np.concatenate(trues, axis=0)
        batchX=np.concatenate(batchX, axis=0)
    else:
        preds=preds[0]
        trues=trues[0]
        batchX=batchX[0]
        
    if CompId and folder_path:
        # print(f"batchX.shape:{batchX.shape};trues.shape:{trues.shape};preds.shape:{preds.shape}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        PredData = np.concatenate((batchX[:,:,-1:], preds), axis=1)
        TrueData = np.concatenate((batchX[:,:,-1:], trues), axis=1)
        np.save(f"{folder_path}/{CompId} EntirePredData.npy", PredData)
        np.save(f"{folder_path}/{CompId} EntireTrueData.npy", TrueData)
    
    test_Y = np.array(test_Y)
    while test_Y.ndim < 3:
        test_Y = np.expand_dims(test_Y, axis = -1)
    
    preds = preds.reshape(-1, test_Y.shape[1], test_Y.shape[2])         # (L, time_lag, features)
    trues = trues.reshape(-1, test_Y.shape[1], test_Y.shape[2])
        
    # 转译
    original_preds, original_trues = np.copy(preds), np.copy(trues)
    while test_X.ndim < 3:
        test_X = np.expand_dims(test_X, axis = -1)
    # while original_preds.ndim < 3:
    #     original_preds = np.expand_dims(original_preds, axis = -1)
    # while original_trues.ndim < 3:
    #     original_trues = np.expand_dims(original_trues, axis = -1)
    # while preds.ndim < 3:
    #     preds = np.expand_dims(preds, axis = -1)
    # while trues.ndim < 3:
    #     trues = np.expand_dims(trues, axis = -1)
    
    if preds.shape[-1] == test_X.shape[-1]:
        for i in range(test_Y.shape[1]):
            original_preds[:,i,:] = scaler.inverse_transform(np.copy(preds[:,i,:]))
            original_trues[:,i,:] = scaler.inverse_transform(np.copy(trues[:,i,:]))
    elif preds.shape[-1] == 1:
        for i in range(test_Y.shape[1]):
            adjust_preds = np.repeat(np.copy(preds[:,i,:]), test_X.shape[-1], axis = 1)
            original_adjust_preds = scaler.inverse_transform(adjust_preds)
            original_preds[:,i,:] = np.copy(original_adjust_preds[:,-1].reshape(-1,1))
            adjust_trues = np.repeat(np.copy(trues[:,i,:]), test_X.shape[-1], axis = 1)
            original_adjust_trues = scaler.inverse_transform(adjust_trues)
            original_trues[:,i,:] = np.copy(original_adjust_trues[:,-1].reshape(-1,1))
    
    if args.verbose:
        test_loss = criterion(torch.from_numpy(trues), torch.from_numpy(preds))
        original_loss = criterion(torch.from_numpy(original_trues), torch.from_numpy(original_preds))
        time2 = time.time()
        timepoint = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"Test loss:{test_loss:.4f} | Test loss (Original): {original_loss:.4f} | Cost time: {time2-time1:.2f}s | {timepoint}")
    
    return predictData(preds, trues, original_trues, original_preds, model)
    
           
def TimeSeries_Exp(data, train_data, model, args, target = None, CompId = None, folder_path = None):
    class predictData():
        def __init__(self, train_Outputs, vail_Outputs, test_Outputs):
            self.preds = test_Outputs.preds
            self.trues = test_Outputs.trues
            self.original_preds = test_Outputs.original_preds
            self.original_trues = test_Outputs.original_trues
            self.model = test_Outputs.model
            
            self.train_preds = train_Outputs.original_preds
            self.train_trues = train_Outputs.original_trues
            self.vail_preds = vail_Outputs.original_preds
            self.vail_trues = vail_Outputs.original_trues
            
    train_lag = int(data.shape[0]*args.train_ratio)
    if isinstance(data, np.ndarray):
        if train_data.ndim == 1:
            train_data = np.expand_dims(train_data, axis = -1)
        if data.ndim == 1:
            data = np.expand_dims(data, axis = -1)
        # scaler = StandardScaler().fit(train_data)
        # train_data = scaler.transform(train_data)
        # data = scaler.transform(data)
        if CompId and CompId == "trend":
            train_scaler = MinMaxScaler().fit(train_data)
            scaler = MinMaxScaler().fit(train_data[:,-1:])
        else:
            train_scaler = StandardScaler().fit(train_data)
            scaler = StandardScaler().fit(train_data[:,-1:])
        train_data = train_scaler.transform(train_data)
        data = train_scaler.transform(data)
        
        Train_X, Train_Y = arr_loader(np.copy(train_data), args.pred_len, args.seq_len)
        Test_X, Test_Y = arr_loader(np.copy(data[train_lag-args.seq_len:]), 
                                    args.pred_len, args.seq_len)
    elif isinstance(data, pd.DataFrame):
        # 将目标特征移到最后一列
        cols = list(data.columns)
        cols.remove(target)
        cols = cols + [target]
        data = data[cols].copy()
        train_data = train_data[cols].copy()
        
        # TargetValues = data[target].values
        # data = data.drop(target, axis = 1)
        # data[target] = list(TargetValues)
        # TargetTrainValues = train_data[target].values
        # train_data = train_data.drop(target, axis = 1)
        # train_data[target] = list(TargetTrainValues)
        
        if CompId and CompId == "trend":
            train_scaler = MinMaxScaler().fit(train_data.values)
            scaler = MinMaxScaler().fit(train_data.values[:,-1:])
        else:
            train_scaler = StandardScaler().fit(train_data.values)
            scaler = StandardScaler().fit(train_data.values[:,-1:])
        
        train_data = pd.DataFrame(train_scaler.transform(train_data.values), 
                                  columns=cols)
        data = pd.DataFrame(train_scaler.transform(data.values), 
                            columns=cols)
        
        
        # scaler = StandardScaler().fit(train_dataValues)
        # train_dataValues = scaler.transform(train_dataValues)
        # train_data = pd.DataFrame(train_dataValues, columns=data.columns)
        # dataValues = scaler.transform(dataValues)
        # data = pd.DataFrame(dataValues, columns=data.columns)
        
        TrainData, TestData = train_data.copy().reset_index(drop=True), data.iloc[train_lag-args.seq_len:].copy().reset_index(drop=True)
        Train_X, Train_Y = df_loader(TrainData, target, args.pred_len, args.seq_len)
        Test_X, Test_Y = df_loader(TestData, target, args.pred_len, args.seq_len)
    
    train_split = int(Train_X.shape[0] * (1-args.vali_split))
    if isinstance(model, nn.Module):
        train_Outputs = Modelpredict(model, Train_X[:train_split], Train_Y[:train_split], scaler, args)
        vail_Outputs = Modelpredict(model,Train_X[train_split:], Train_Y[train_split:], scaler, args)
        test_Outputs = Modelpredict(model, Test_X, Test_Y, scaler, args, CompId, folder_path)
    else:    
        # 模型定义
        args.input_size = data.shape[1]
        model = model(args)
        
        # 模型训练
        model = ModelTrain(model, Train_X, Train_Y, args)
        
        # 模型评估
        train_Outputs = Modelpredict(model, Train_X[:train_split], Train_Y[:train_split], scaler, args)
        vail_Outputs = Modelpredict(model,Train_X[train_split:], Train_Y[train_split:], scaler, args)
        test_Outputs = Modelpredict(model, Test_X, Test_Y, scaler, args, CompId, folder_path)
    return predictData(train_Outputs=train_Outputs, vail_Outputs=vail_Outputs, test_Outputs=test_Outputs)
    
    

        
            
            
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    