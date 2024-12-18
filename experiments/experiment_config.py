class ExperimentConfig():
    def __init__(self,
                 num_layers:int,
                 forward_dim:int,
                 dropout:float,
                 learning_rate:float,
                 batch_size:int,
                 num_heads:int = 8,
                 emb_dim:int = 128,
                 optimizer:str = "AdamW",
                 grad_clip:int = 1,
                 batch_size_eval:int = 500,
                 epochs:int = 40,
                 evaluation_interval:int = -1,
                 model_save_interval:int = -1,
                 device:str = "cuda",
                 model_save_path:str ="saved_data/models/",
                 tensorboard_log_path:str ="saved_data/tensorboard_runs/",
                 results_dict_path:str ="saved_data/eval_results/",
                 model_weights_path:str = None,
                 train_model:bool = True,
                 use_tensorboard=True,
                 experiment_name:str = None,
                 k_fold:int|tuple[int,int] = None, # k to run k-fold cross validation, (k,n) to run fold n of k
                 detailed_logging:bool = True,
                 overwrite_if_saving = False
                 ):
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.forward_dim = forward_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.grad_clip = grad_clip
        self.optimizer = optimizer
        self.epochs = epochs
        self.model_save_path = model_save_path
        self.tensorboard_log_path = tensorboard_log_path
        self.results_dict_path = results_dict_path
        self.model_weights_path = model_weights_path
        self.device = device
        self.train_model = train_model
        self.use_tensorboard = use_tensorboard
        self.experiment_name = experiment_name
        self.evaluation_interval = evaluation_interval
        self.model_save_interval = model_save_interval
        self.k_fold = k_fold # (k, n) where k is the number of folds and n is the fold number
        self.detailed_logging = detailed_logging
        self.overwrite_if_saving = overwrite_if_saving
        if k_fold:
            self.model_save_path = model_save_path+"k_fold/"
            self.tensorboard_log_path = tensorboard_log_path+"k_fold/"
            self.results_dict_path = results_dict_path+"k_fold/"

        if self.evaluation_interval == -1:
            self.evaluation_interval = self.epochs
        
        if self.model_save_interval == -1:
            self.model_save_interval = self.epochs
        
        

    def __str__(self):
        s = ""
        s+= "-----HYPER PARAMS-----\n"
        s += "emb_dim: "
        s += "emb_dim: " + str(self.emb_dim) + "\n"
        s += "num_layers: " + str(self.num_layers) + "\n"
        s += "num_heads: " + str(self.num_heads) + "\n"
        s += "forward_dim: " + str(self.forward_dim) + "\n"
        s += "dropout: " + str(self.dropout) + "\n"
        s += "learning_rate: " + str(self.learning_rate) + "\n"
        s += "batch_size: " + str(self.batch_size) + "\n"
        s += "batch_size_eval: " + str(self.batch_size_eval) + "\n"
        s += "grad_clip: " + str(self.grad_clip) + "\n"
        s += "optimizer: " + str(self.optimizer) + "\n"
        s += "epochs: " + str(self.epochs) + "\n"
        s+= "--------PATHS---------\n"
        s += "model_save_path: " + str(self.model_save_path) + "\n"
        s += "tensorboard_log_path: " + str(self.tensorboard_log_path) + "\n"
        s += "results_dict_path: " + str(self.results_dict_path) + "\n"
        s += "model_weights_path: " + str(self.model_weights_path) + "\n"
        s+= "--------OTHER---------\n"
        if not self.experiment_name:
            s += "experiment_name: AUTO\n"
        else:
            s += "experiment_name: " + str(self.experiment_name) + "\n"
        s += "evaluation_interval: " + str(self.evaluation_interval) + "\n"
        s += "model_save_interval: " + str(self.model_save_interval) + "\n"
        s += "k_fold: " + str(self.k_fold) + "\n"
        s += "use_tensorboard: " + str(self.use_tensorboard) + "\n"
        s += "detailed_logging: " + str(self.detailed_logging) + "\n"
        s += "train_model: " + str(self.train_model) + "\n"
        s += "overwrite_if_saving: " + str(self.overwrite_if_saving) + "\n"
        s += "device: " + str(self.device) + "\n"
        return s
    
    def __repr__(self):
        return self.__str__()

class E1_Config(ExperimentConfig):
    # need kwargs
    def __init__(self, **kwargs):
        super().__init__(num_layers=1,
                forward_dim=512,
                dropout=0.05,
                learning_rate=7e-4,
                batch_size=64,
                model_save_path="saved_data/models/E_1/",
                results_dict_path="saved_data/eval_results/E_1/",
                tensorboard_log_path="saved_data/tensorboard_runs/E_1/",
                # pass kwars to super
                **kwargs,
                )
    
class E2_Config(ExperimentConfig):
    def __init__(self, **kwargs):
        super().__init__(num_layers=2,
                forward_dim=256,
                dropout=0.15,
                learning_rate=2e-4,
                batch_size=16,
                model_save_path="saved_data/models/E_2/",
                results_dict_path="saved_data/eval_results/E_2/",
                tensorboard_log_path="saved_data/tensorboard_runs/E_2/",
                # pass kwars to super
                **kwargs,
                )

class E3_Config(ExperimentConfig):
    def __init__(self, **kwargs):
        super().__init__(num_layers=2,
                forward_dim=256,
                dropout=0.15,
                learning_rate=2e-4,
                batch_size=16,
                model_save_path="saved_data/models/E_3/",
                results_dict_path="saved_data/eval_results/E_3/",
                tensorboard_log_path="saved_data/tensorboard_runs/E_3/",
                # pass kwars to super
                **kwargs,
                )
