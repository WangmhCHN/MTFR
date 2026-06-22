import torch.nn as nn
import torch
import math

class obtain_predictor:
    def __init__(self, configs):
        if configs.model == "GRU":
            self.preditor = GRUNN


    
# GRU -------------------------------------------------------------------------
class GRUNN(nn.Module):
    def __init__(self, args):
        super(GRUNN, self).__init__()
        
        self.gru_dim = args.GRU_units
        self.linear_dim = args.GRU_units
        
        # GRU layer
        self.gru = nn.GRU(args.input_size, self.gru_dim, batch_first=True, 
                          num_layers=args.num_layers, bias=True)
        
        # FC layer
        self.fc1 = nn.Linear(self.linear_dim, args.Linear_units, bias=True)
        self.fc2 = nn.Linear(args.Linear_units, args.pred_len, bias=True)
        
        
    def forward(self, x):
        # GRU 层
        gru_out, _ = self.gru(x)                    # GRU 的输出, shape = [batch, seq_len, gru_dim]
        # print(f"gru_out.shape = {gru_out.shape}. batch = {x.size(0)}, linear_dim={self.linear_dim}")
        # gru_out = gru_out.reshape(-1, 1, self.linear_dim)   # shape = [batch, 1, seq_len*gru_dim]
        
        # 全连接层
        out = nn.Tanh()(self.fc1(gru_out[:,-1:,:]))
        out = self.fc2(out)
        out = out.view(out.size(0), -1, 1)
        return out
        
