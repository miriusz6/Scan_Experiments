# OTHER
import os 
from tensorboardX import SummaryWriter
from tqdm.notebook import tqdm
import pickle
import json
from math import ceil
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
                mk_model = None,
                mk_train_dataset = None,
                mk_test_dataset = None,
                evaluate_model = None
                ):
        self.e_type = experiment_type
        self.config = config
        self.name = self.e_type.name if self.config.experiment_name is None else self.config.experiment_name
        if ("rep") not in self.name:
            self.name = self.name + "_rep_1"
        
        self._mk_model = (lambda: mk_model) if mk_model is not None else self._mk_base_transformer_model
        self._mk_train_dataset = (lambda: mk_train_dataset) if mk_train_dataset is not None else self._mk_train_dataset_def
        self._mk_test_dataset = (lambda: mk_test_dataset) if mk_test_dataset is not None else self._mk_test_dataset_def
        self.evaluate_model = evaluate_model
        
        self.device = config.device
        self.writer = None
        self.train_dataset = self._mk_train_dataset()
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.result_container:EvaluationResultContainer = EvaluationResultContainer()
             
        self.use_k_fold = self.config.k_fold is not None
        self.current_fold = None
        self.folds = None
        self.model = None
        
         
        self.overwrite = self.config.overwrite_if_saving
        
        self.use_TB = config.use_tensorboard

        self.epoch_checkpoint = 0
       

    def _validate_path(self, path):
        path_alter = os.path.join(CURR_DIR, path)
        if os.path.exists(path):
            return path
        elif os.path.exists(path_alter):
            return path_alter
        # create path if not exists
        if not os.path.exists(path):
            os.makedirs(path)
            print("Path created: ", path)
            return path
        
    
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

            reps_indx = full_path.find('rep_')
            i = 0
            while f_dir_exists(alter_path):
                i += 1
                # _rep is always added
                if reps_indx != -1:
                    curr_rep = full_path[reps_indx+4:]
                    if ext != "":
                        curr_rep = curr_rep.split('.')[0]
                    curr_rep = int(curr_rep)
                    alter_path = full_path[:reps_indx+4] + str(curr_rep+i) + ext
                # fail safe in case name was modified after initialisation
                else:
                    alter_path = os.path.join(path, f"{self.name}({i})"+ext)
                
            full_path = alter_path
            new_name = full_path.split('/')[-1]
            if ext != "":
                new_name = new_name.split('.')[0]
            
            self.name = new_name
        if overwrite:
            if f_dir_exists(full_path):
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

    def _mk_train_dataset_def(self):
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
    
    def _mk_test_dataset_def(self):
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
         # 

    def _mk_dataloaders(self): 

        train_loader = DataLoader(self.train_dataset,
                                batch_size=self.config.batch_size,
                                shuffle=True
                                )
        test_loader = DataLoader(self.test_dataset,
                                batch_size=self.config.batch_size_eval,
                                )
        return train_loader, test_loader

    def _mk_base_transformer_model(self):
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

    def _load_weights(self,p = ""):
        if p == "":
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
            # use tqdm
            fold_bar = tqdm(range(self.folds), desc="FOLD")
            for f_num in fold_bar:
                self.current_fold = f_num
                self.train_dataset = self._mk_train_dataset()
                self._train()

    def _train(self):
        r_indx = self.name.find("_rep")
        org_name_base,org_rep_info = self.name[:r_indx], self.name[r_indx:]

        fold_info = ""
        tb_fold_info = ""
        if self.use_k_fold:
            fold_info = f"_fold_{self.current_fold+1}_of_{self.folds}"
            tb_fold_info = f"/fold{self.current_fold+1}"
        self.name = org_name_base + fold_info + org_rep_info

        if self.use_TB:
            self._validate_path(self.config.tensorboard_log_path)
            self.tensorboard_path = self._mk_path(self.config.tensorboard_log_path,self.overwrite)
            self.writer = SummaryWriter(self.tensorboard_path)
        self._validate_path(self.config.results_dict_path)
        self._validate_path(self.config.model_save_path)
        

        self._prepate_data()
        self.model = self._mk_model()
        if self.config.model_weights_path is not None:
            self._load_weights()
        if self.epoch_checkpoint != 0:
            # f"[resume_{self.epoch_checkpoint}_epoch]"
            n = org_name_base.split("{")[0] + org_name_base.split("}")[1]
            p = self.config.model_save_path + n
            p += fold_info + f"_epoch_{self.epoch_checkpoint }" + org_rep_info
            p += ".weights"
            self._load_weights(p)

        criterion = CrossEntropyLoss(ignore_index=self.train_dataset.vocab.pad_idx)
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.00001,
        )
        grad_clip = self.config.grad_clip
        batch_size = self.config.batch_size
        
        max_epoch = self.config.epochs if self.config.epochs is not None else ceil(self.config.max_steps/len(self.train_dataset))
        max_steps = self.config.max_steps if self.config.max_steps is not None else (max_epoch*len(self.train_loader)*batch_size)+1
        max_batches = len(self.train_loader)*max_epoch
        batch_num = 0
        step_num = 0
        batch_step = 0

        eval_interval = self.config.evaluation_interval
        model_sv_interval = self.config.model_save_interval

        

        if self.config.detailed_logging:
            print("Training started for experiment: ", self.e_type.name)#, fold_info)

   
        # TRAINING LOOP
        epoch_progress = tqdm(range(self.epoch_checkpoint, max_epoch), desc="EPOCH"+fold_info)
        for epoch in epoch_progress:
            if max_steps is not None and step_num >= max_steps:
                    print("Early Stopping: Max steps reached")
                    break
            total_loss = 0
            self.model.train()
            #batch_bar = tqdm(self.train_loader, desc="BATCH")
            #for batch in batch_bar:
            batch_step = 0
            for batch in self.train_loader:
                if max_steps is not None and step_num >= max_steps:
                    break


                inputs, decoder_inputs, target_label_indices = batch
                optimizer.zero_grad()
                #is_teacher = True 
                is_teacher = batch_step % 2 == 0
                
                if is_teacher:
                    out = self.model(inputs, decoder_inputs)
                else:
                    with torch.no_grad():
                        self.model.eval()
                        fst_out = self.model(inputs, decoder_inputs)
                        fst_out = fst_out.argmax(2)
                    self.model.train()

                    inp_msks = self.model.create_src_mask(inputs)
                    enc_out = self.model.encoder(inputs, inp_msks)

                    tgt_msks = self.model.create_tgt_mask(fst_out)
                    dec_out = self.model.decoder(fst_out, enc_out, inp_msks, tgt_msks)
                    out = dec_out

                

                loss = criterion(out.permute(0, 2, 1), target_label_indices)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()

                total_loss += loss.item()

                batch_num += 1
                batch_step += 1
                step_num += batch_size

                if self.config.detailed_logging or batch_num == len(self.train_loader):
                    print(f"Epoch {epoch+1}/{max_epoch} Batch {batch_num}/{max_batches} Trining Loss: {total_loss / (batch_num + 1)} Teacher: {is_teacher} Step: {step_num}/{max_steps}" )

                

            epoch += 1
            if self.config.detailed_logging or epoch == max_epoch:
                print(f"Epoch {epoch}/{max_epoch} Batch {batch_num}/{max_batches} Trining Loss: {total_loss / (batch_num + 1)}")

            # EVALUATION 
            if epoch % eval_interval == 0 or epoch == max_epoch or max_steps is not None and step_num >= max_steps:
                self.model.eval()
                result = evaluate_model_batchwise(self.model,self.test_dataset, self.test_loader, self.test_dataset.vocab, self.device)
                #result = EvaluationResult(result, self.name+fold_info+f"_epoch_{epoch}", self.e_type)
                result = EvaluationResult(result, self.name+f"_epoch_{epoch}", self.e_type)
                if self.config.detailed_logging or epoch == max_epoch:
                    result.print()
                self.result_container.append_results(result)
            
            if epoch % model_sv_interval == 0 or epoch == max_epoch:
                buff = self.name
                self.name = org_name_base + fold_info + f"_epoch_{epoch}" + org_rep_info
                self.save_model()
                self.name = buff


            if self.use_TB:
                self.writer.add_scalar(tag = 'TrainLoss'+tb_fold_info,
                                    scalar_value = total_loss,
                                    global_step = epoch)
            
        if self.use_TB:
            self.writer.close()
        

        self.name = org_name_base + org_rep_info
            
    def save_model(self, path = ""):
        overwrite = self.overwrite
        if path == "":
            path = self._mk_path(self.config.model_save_path,overwrite,ext=".weights")
        print("Model saved at: ", path)
        torch.save(self.model.state_dict(),path)
    
    def save_results(self, path = ""):
        overwrite = self.overwrite
        if path == "":
            path = self._mk_path(self.config.results_dict_path,overwrite,ext=".json")
        else:
            path = self._mk_path(path, overwrite)

        print("Evaluation results will be saved at: ", path)
        d = EvaluationResultContainer.to_dict(self.result_container)
        with open(path, "w") as f:
            json.dump(d,f,indent=4)
            
    def run(self, from_epoch = 0):
        print("-"*50)
        self.epoch_checkpoint = from_epoch
        if self.epoch_checkpoint != 0:
            r_indx = self.name.find("_rep")
            base,org_rep_info = self.name[:r_indx], self.name[r_indx:]
            self.name = base + '{' + f"resume_{self.epoch_checkpoint}_epoch" + '}'+ org_rep_info
        if not self.config.train_model:
            print("Enable model training. This is a debug mode")
        if self.use_k_fold:
            self._k_fold_train()
        else:
            self._train()
            print("Training finished for experiment: ", self.e_type.name)
            if self.use_TB and self.config.detailed_logging:
                print("Tensorboard logs saved at: ", self.tensorboard_path)
        
        return self.result_container
        



