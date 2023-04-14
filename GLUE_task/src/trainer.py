########## The following part is copied from Transformers' trainer (3.4.0) ########## 

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
import os
from typing import Dict, Optional
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

import transformers
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import TrainOutput
from transformers.utils import logging
from tqdm import tqdm, trange
import numpy as np
import faiss
import pickle


logger = logging.get_logger(__name__)

########## The above part is copied from Transformers' trainer (3.4.0) ########## 

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]
 
    raise Exception("No metric founded for {}".format(metrics))

class Trainer(transformers.Trainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps 
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        model = self.model

        # Train
        total_train_batch_size = (self.args.train_batch_size * self.args.gradient_accumulation_steps)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch"
        )

        for epoch in train_iterator:
            # for semi-supervised-learning: generate pesudo labels by vanilla-roberta, each epoch
            if self.args.ssl:
                ssl_dataloader = self.get_eval_dataloader(self.train_dataset) # dont shuffle
                pesudo_labels = self.get_ssl_pseudo_labels(self.model, ssl_dataloader)
                pesudo_dataset = self.set_ssl_pseudo_labels(self.train_dataset, pesudo_labels)
                self.train_dataset = pesudo_dataset
                train_dataloader = self.get_train_dataloader()

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
            # init for knn saveing mask features
            # if self.args.train_with_knn and epoch == 0: # don't update datastore during training
            if self.args.train_with_knn:
                train_dataloader = self.get_train_dataloader()
                self.get_mask_features_for_knn(self.model, train_dataloader)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self.training_step(model, inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["norm"] = norm.item()
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = scheduler.get_last_lr()[0]
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)
                    
                    metrics = None
                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                        output = self.evaluate()
                        metrics = output.metrics
                        objective = self.dev_objective(metrics)
                        if objective > self.objective:
                            logger.info("Best dev result: {}".format(objective))
                            self.objective = objective
                            self.save_model(self.args.output_dir) 

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            self.clear_mask_features()  # if don't update datastore during training, it should be comment out
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step, {'metric': self.objective})


    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        if self.args.only_train_knn:
            self.model.model_args.knn_infer = False

        # if self.model.model_args.knn_infer and eval_dataset is not None:    # don't update dataset during trianing and eval
        if self.model.model_args.knn_infer:
            self.clear_mask_features()
            train_dataloader = self.get_train_dataloader()
            self.get_mask_features_for_knn(self.model, train_dataloader)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")
        
        self.log(output.metrics)
        
        self.model.model_args.knn_infer = self.args.knn_infer

        return output

    def get_mask_features_for_knn(self, model, dataloader):
        # logger.info("**** Getting mask features for KNN ****")
        with torch.no_grad():
            # for inputs in tqdm(dataloader, total=len(dataloader), desc='KNN'):
            for inputs in dataloader:
                model.eval()
                inputs = self._prepare_inputs(inputs)
                model(**inputs, save_mask=True)
                
            model.total_features = np.concatenate(model.total_features, axis=0)
            # norm features for consine similarity
            if self.args.sim_metric == 'consine':
                faiss.normalize_L2(model.total_features.astype(np.float32))
            model.mask_features.add(model.total_features)
        
    def clear_mask_features(self):
        self.model.total_features = []
        self.model.mask_features = faiss.IndexFlatL2(self.model.config.hidden_size)
        self.model.maskid2labelid = {}
        self.model.cnt_batch = 0
        
    def get_ssl_pseudo_labels(self, model, dataloader):
        pseudo_labels = []
        with torch.no_grad():
            for inputs in dataloader:
                model.eval()
                inputs = self._prepare_inputs(inputs)
                logits = model(**inputs, is_ssl=True)
                preds = torch.argmax(logits, dim=-1)
                pseudo_labels.extend(preds.detach().cpu().tolist())
        return pseudo_labels
        
    def set_ssl_pseudo_labels(self, dataset, pesudo_labels):
        for i in range(len(dataset.query_examples)):
            dataset.query_examples[i].label = pesudo_labels[i]          # for sst-5!!
        return dataset


    def save_mask_features(self):
        print(faiss.MatrixStats(self.model.total_features).comments)
        faiss.write_index(self.model.mask_features, os.path.join(self.args.output_dir, "faiss_mask_features.index"))
        with open(os.path.join(self.args.output_dir, "maskid2labelid.pkl") ,'wb') as file:
            pickle.dump(self.model.maskid2labelid, file)
        print(f"number of  entity embedding : {len(self.model.maskid2labelid)}")

    def load_mask_features(self):
        self.model.maskid2labelid = pickle.load(open(os.path.join(self.args.output_dir, "maskid2labelid.pkl") ,'rb'))
        self.model.mask_features = faiss.read_index(os.path.join(self.args.output_dir, "faiss_mask_features.index"))
        print('Reading successful!')
