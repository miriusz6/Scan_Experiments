# OTHER
import os 
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pickle
# TORCH
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
# LOCAL
from transformer import Transformer
from dataset.scan_dataset import ScanDataset
from experiments.experiment_type import ExperimentType
from experiments.experiment_config import ExperimentConfig
from experiments.evaluation import evaluate_model_batchwise
from experiments.evaluation_result import EvaluationResult
from experiments.evaluation_result_container import EvaluationResultContainer
# const
CURR_DIR = os.path.dirname(os.path.realpath(__file__))

class Experiment():
    def __init__(self, 
                experiment_type: ExperimentType,
                config: ExperimentConfig,
                ):
        self.e_type = experiment_type
        self.config = config
        self.name = self.e_type.name if self.config.experiment_name is None else self.config.experiment_name
        self.device = config.device
        
        self.train_dataset = self._mk_train_dataset()
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.result_container:EvaluationResultContainer = EvaluationResultContainer()

        self.use_k_fold = self.config.k_fold is not None
        self.current_fold = None
        self.folds = None
        self.model = None
        
        self.use_TB = config.use_tensorboard
        if self.use_TB:
            self.tensorboard_path = self._mk_path(config.tensorboard_log_path,True)
        else:
            self.writer = None

    def _validate_path(self, path):
        path_alter = os.path.join(CURR_DIR, path)
        if os.path.exists(path):
            return path
        elif os.path.exists(path_alter):
            return path_alter
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    def _validate_file(self, path):
        path_alter = os.path.join(CURR_DIR, path)
        if os.path.isfile(path):
            return path
        elif os.path.isfile(path_alter):
            return path_alter
        raise FileNotFoundError(f"File does not exist: {path}")

    def _mk_path(self, raw_p, overwrite=False, ext=""):
        path = self._validate_path(raw_p)
        full_path = os.path.join(path, self.name)+ext
        f_dir_exists = lambda p: os.path.isfile(p) or os.path.isdir(p)
        if  f_dir_exists(full_path) and not overwrite:
            # try path/name(1), path/name(2), ...
            alter_path = full_path
            i = -1
            while f_dir_exists(alter_path):
                i += 1
                alter_path = os.path.join(path, f"{self.name}({i})"+ext)
                
            full_path = alter_path
            self.name = f"{self.name}({i})"
        if overwrite:
            print("The following file will be overwritten:", full_path)
            if os.path.exists(full_path):
                os.remove(full_path)
        return full_path

    def _prepate_data(self):
        if self.use_k_fold:
            self.train_dataset, self.test_dataset = self.train_dataset.pull_nth_fold_out(self.folds, self.current_fold)
        else:
            self.test_dataset = self._mk_test_dataset()
        self.train_loader, self.test_loader = self._mk_dataloaders()

    def _mk_train_dataset(self):
        paths = self.e_type.get_data_paths()
        max_len = self.config.emb_dim
        device = self.device
        train_dataset = ScanDataset(
            dataset_path= paths["train"],
            in_seq_len=max_len,
            out_seq_len=max_len + 20,
            device=device,
        )
        return train_dataset
    
    def _mk_test_dataset(self):
        # call alwyas after mk_train_dataset
        paths = self.e_type.get_data_paths()
        max_len = self.config.emb_dim
        device = self.device
        test_dataset = ScanDataset(
            dataset_path= paths["test"],
            vocab=self.train_dataset.vocab,
            in_seq_len=max_len,
            out_seq_len=max_len + 20,
            device=device,
        )
        return test_dataset
    
    def _mk_dataloaders(self):
        train_loader = DataLoader(self.train_dataset,
                                batch_size=self.config.batch_size,
                                shuffle=True
                                )
        test_loader = DataLoader(self.test_dataset,
                                batch_size=self.config.batch_size_eval,
                                )
        return train_loader, test_loader

    def _mk_model(self):
        model = Transformer(
            src_vocab_size=len(self.train_dataset.vocab),
            tgt_vocab_size=len(self.train_dataset.vocab),
            src_pad_idx=self.train_dataset.vocab.pad_idx,
            tgt_pad_idx=self.train_dataset.vocab.pad_idx,
            dropout=self.config.dropout,
            emb_dim=self.config.emb_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            forward_dim=self.config.forward_dim,
            max_len=self.config.emb_dim + 20,
        ).to(self.device)
        return model

    def _load_weights(self):
        p = self.config.model_weights_path
        p = self._validate_file(p)
        print("Loading model weights from: ", p)
        self.model.load_state_dict(torch.load(p))

    def _k_fold_train(self):
        if isinstance(self.config.k_fold, tuple):
            self.folds = self.config.k_fold[0]
            self.current_fold = self.config.k_fold[1]
            self._train()
        else:
            self.folds = self.config.k_fold
            for i in range(self.config.k_fold):
                self.current_fold = i
                self.train_dataset = self._mk_train_dataset()
                self._train()

    def _train(self):
        if self.use_TB:
            self.writer = SummaryWriter(self.tensorboard_path)
        self._prepate_data()
        self.model = self._mk_model()
        if self.config.model_weights_path is not None:
            self._load_weights()
        criterion = CrossEntropyLoss(ignore_index=self.train_dataset.vocab.pad_idx)
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.00001,
        )
        grad_clip = self.config.grad_clip
        batch_size = self.config.batch_size
        max_epoch = self.config.epochs
        max_batches = len(self.train_loader)*max_epoch
        batch_num = 0
        step_num = 0
        step = 0

        eval_interval = self.config.evaluation_interval

        fold_info = ""
        if self.use_k_fold:
            fold_info = f"_fold_{self.current_fold+1}/{self.folds}"

        if self.config.detailed_logging:
            print("Training started for experiment: ", self.e_type.name, fold_info)

        # TRAINING LOOP
        epoch_progress = tqdm(range(max_epoch), desc="EPOCH"+fold_info)
        for epoch in epoch_progress:
            total_loss = 0
            self.model.train()
            for batch in self.train_loader:
                inputs, decoder_inputs, target_label_indices = batch

                optimizer.zero_grad()
                out = self.model(inputs, decoder_inputs)
                loss = criterion(out.permute(0, 2, 1), target_label_indices)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()

                total_loss += loss.item()
                    
                batch_num += 1
                step_num += batch_size

            epoch += 1
            if self.config.detailed_logging or epoch == max_epoch:
                print(f"Epoch {epoch}/{max_epoch} Batch {batch_num}/{max_batches} Trining Loss: {total_loss / (step + 1)}")

            # EVALUATION 
            if epoch % eval_interval == 0 or epoch == max_epoch:
                self.model.eval()
                result = evaluate_model_batchwise(self.model,self.test_dataset, self.test_loader, self.test_dataset.vocab, self.device)
                result = EvaluationResult(result, self.name+fold_info+f"_epoch_{epoch}", self.e_type)
                if self.config.detailed_logging or epoch == max_epoch:
                    result.print()
                self.result_container.append_results(result)
            
            if self.use_TB:
                self.writer.add_scalar(tag = 'Loss/train',
                                    scalar_value = total_loss,
                                    global_step = epoch)
            
        if self.use_TB:
            self.writer.close()
        
        if self.config.detailed_logging:
            print("Training finished for experiment: ", self.e_type.name)
            
    def save_model(self, path = "", overwrite = False):
        if path == "":
            path = self._mk_path(self.config.model_path,overwrite,ext=".weights")
        else:
            path = self._mk_path(path, overwrite)
        print("Model will be saved at: ", path)
        torch.save(self.model.state_dict(),path)
    
    def save_results(self, path = "", overwrite = False):
        if path == "":
            path = self._mk_path(self.config.results_dict_path,overwrite,ext=".data")
        else:
            path = self._mk_path(path, overwrite)

        print("Evaluation results will be saved at: ", path)
        with open(path, "wb") as f:
            pickle.dump(self.result_container, f)

    def run(self):
        print("-"*50)
        if self.use_TB and self.config.detailed_logging:
            print("Tensorboard logs will be saved at: ", self.tensorboard_path)
        if self.config.train_model:
            if self.use_k_fold:
                self._k_fold_train()
            else:
                self._train()
        return self.result_container
        



