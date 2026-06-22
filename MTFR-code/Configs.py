import torch

from Decomposition import SSTL_UFFT, SSTL_MWT
from FeatureSelection import PI, DeepSHAP, ALLFeatures

# The parameters of train experiment
class Paras():
    def __init__(self):
        # basic config
        self.model = None                       # GRU
        self.seq_len = 0
        self.pred_len = 0
        self.data_name = None
        self.fored_excute = 0
        self.period = 1                         # the maximum periodic value
        self.periods = 1                        # the list of periods
        
        # data loader
        self.train_ratio = 0.8
        
        # decomposer
        self.resid_decompNum = 1
        self.decompMethod = None
        self.wavelet = "db10"          # db2, db4, db10, symlets, coiflets
        
        # feature selection
        self.select_features_n = 15         # 最大选择的特征数
        self.FSMethod = "PI"                # PI/DeepSHAP/ALL
        self.corr_threshold = 0.9           # 去除相关性冗余的阈值
        self.eval_dataset_type = "train"    # 使用哪种数据集评估特征重要性. 训练集: 'train'; 验证集: 'vali'
        self.n_pi_repeats = 10              # permutation_importance方法的扰动数
        self.score_folder = r"D:/WangMH_SYSU/PyProject/Project/MTFR3/result/Score"
        self.score_method = None            # 结构：{FSMethod}-{FS_Eval_datatype}-{current_decomp_method}-{PredictModel}
        self.FeatureSelection_logger = None
        self.max_class_n = 10               # 允许提取的最大类别个数
        self.score_id = None                # 哪一个部分的数据
        
        # training 
        self.vali_split = 0.2                    # 验证集比例
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 0.001
        self.isEarlyStopping = True
        self.lradj = False
        self.lradj_time = 100
        self.verbose = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_logger = None
        self.patience = 3
        
        # model ------------------------------
        self.input_size = 0
        # GRU
        self.GRU_units = 50
        self.Linear_units = 10
        self.num_layers = 1
        
        
        
class obtain_decomposer:
    def __init__(self, decompMethod, wavelet = "db10"):
        self.decompMethod = decompMethod
        self.wavelet = wavelet
        
        self._obtain()
    
    def _obtain(self):
        if self.decompMethod ==  "SSTL_UGDFT":
            self.decomp_method = SSTL_UFFT.decomposer().decomp
            self.decompMethod_name = self.decompMethod
        elif self.decompMethod == "SSTL_MWT":
            self.decomp_method = SSTL_MWT.decomposer(self.wavelet).decomp
            self.decompMethod_name = f"{self.decompMethod}({self.wavelet})"
        else:
            print("The decomposition method is illegal!")
        

class obtain_feature_selector:
    def __init__(self, configs):
        if configs.FSMethod == "PI":
            self.selector = PI.Feature_Selection
        elif configs.FSMethod == "DeepSHAP":
            self.selector = DeepSHAP.Feature_Selection
        elif configs.FSMethod == "ALL":
            self.selector = ALLFeatures.Feature_Selection
            
        
        
        
        
        
        
        
        
        
        
    