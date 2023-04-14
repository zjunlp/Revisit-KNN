import json
import torch
import torch.nn as nn
import numpy as np

from .base import BaseLitModel
from .util import f1_eval, f1_score
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
from functools import partial
from copy import deepcopy
import faiss
from .util import knnLoss
from tqdm import tqdm


def mask_hook(grad_input, st, ed):
    mask = torch.zeros((grad_input.shape[0], 1)).type_as(grad_input)
    mask[st: ed] += 1.0
    # for the speaker unused token12
    mask[1:3] += 1.0
    return grad_input * mask

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


class BertLitModel(BaseLitModel):
    """
    use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """
    def __init__(self, model, args, tokenizer, datamodule):
        super().__init__(model, args)
        self.tokenizer = tokenizer
        self.datamodule = datamodule
        
        with open(f"{args.data_dir}/rel2id.json","r") as file:
            rel2id = json.load(file)
        self.rel2id = rel2id
        
        Na_num = 0
        for k, v in rel2id.items():
            if k == "NA" or k == "no_relation" or k == "Other":
                Na_num = v
                break
        num_relation = len(rel2id)
        # init loss function
        self.loss_fn = multilabel_categorical_crossentropy if "dialogue" in args.data_dir else nn.CrossEntropyLoss()
        # if args.train_with_knn:
        #     self.loss_fn = knnLoss()
        # ignore the no_relation class to compute the f1 score
        self.eval_fn = f1_eval if "dialogue" in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        self.best_f1 = 0
        self.t_lambda = args.t_lambda
        
        self.label_st_id = tokenizer("[class1]", add_special_tokens=False)['input_ids'][0]
        self.tokenizer = tokenizer
    
        self._init_label_word()
        
        # For saving [MASK]
        self.cnt_batch = 0  # record current batch
        self.maskid2labelid = {}
        self.clsid2labelid = {}
        d, measure = self.model.config.hidden_size, faiss.METRIC_L2
        if self.args.ft_with_knn:
            self.cls_features = faiss.IndexFlatL2(d)
        else:
            self.mask_features = faiss.IndexFlatL2(d)
        self.total_features = []


    def _init_label_word(self, ):
        args = self.args
        # ./dataset/dataset_name
        dataset_name = args.data_dir.split("/")[7]
        model_name_or_path = args.model_name_or_path.split("/")[-1]
        label_path = f"dataset/{model_name_or_path}_{dataset_name}.pt"
        # [num_labels, num_tokens], ignore the unanswerable
        if "dialogue" in args.data_dir:
            label_word_idx = torch.load(label_path)[:-1]
        else:
            label_word_idx = torch.load(label_path)
        
        self.num_labels = len(label_word_idx)
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(1, self.num_labels+1)], add_special_tokens=False)['input_ids']]
            
            # for abaltion study
            if self.args.init_answer_words:
                if self.args.init_answer_words_by_one_token:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = word_embeddings.weight[idx][-1]
                else:
                    for i, idx in enumerate(label_word_idx):
                        word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx], dim=0)
                # word_embeddings.weight[continous_label_word[i]] = self.relation_embedding[i]
            
            if self.args.init_type_words:
                so_word = [a[0] for a in self.tokenizer(["[obj]","[sub]"], add_special_tokens=False)['input_ids']]
                meaning_word = [a[0] for a in self.tokenizer(["person","organization", "location", "date", "country"], add_special_tokens=False)['input_ids']]
            
                for i, idx in enumerate(so_word):
                    word_embeddings.weight[so_word[i]] = torch.mean(word_embeddings.weight[meaning_word], dim=0)
            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.model.get_input_embeddings().weight, self.model.get_output_embeddings().weight)
        
        self.word2label = continous_label_word # a continous list
            
                
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, save_mask=False, save_cls=False, is_ssl=False):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, so = batch
        if self.args.ft_with_knn:
            result, cls_features = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        else:
            result, lm_features = self.model(input_ids, attention_mask, return_dict=True, output_hidden_states=True)
        
        if save_mask:   # save mask features
            bsz = input_ids.size(0)
            _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            lm_features = lm_features[torch.arange(bsz), mask_idx]
            self.total_features.append(lm_features.cpu().detach())
            for idx, label in zip(range(bsz), labels):
                self.maskid2labelid[idx + self.cnt_batch] = label.cpu().detach()
            self.cnt_batch += bsz
            return None

        if save_cls:
            bsz = input_ids.size(0)
            self.total_features.append(cls_features.cpu().detach())
            for idx, label in zip(range(bsz), labels):
                self.clsid2labelid[idx + self.cnt_batch] = label.cpu().detach()
            self.cnt_batch += bsz
            return None
        
        logits = result.logits
        if self.args.ft_with_knn:
            logits, knn_logits, combine_logits = self.pvp(logits, input_ids, mode='train', is_ssl=is_ssl, cls_features=cls_features)
        else:
            logits, knn_logits, combine_logits = self.pvp(logits, input_ids, mode='train', is_ssl=is_ssl, lm_features=lm_features)
        if self.args.train_with_knn and not is_ssl:
            loss_fct = knnLoss()
            loss = loss_fct(logits, knn_logits, labels, self.args.alpha)
            # loss = self.loss_fn(logits, knn_logits, labels, self.args.alpha)
        else:
            loss = self.loss_fn(logits, labels)
        self.log("Train/loss", loss)
        return {'loss': loss, 'logits': logits.detach()}
    
    def on_train_epoch_start(self):
        if self.args.train_with_knn:
            train_dataloader = self.datamodule.train_dataloader()
            if self.args.ft_with_knn:
                self.get_cls_features_for_knn(train_dataloader)
            else:
                self.get_mask_features_for_knn(train_dataloader)
        
    def on_training_epoch_end(self):
        if self.args.ft_with_knn:
            self.clear_cls_features()
        else:
            self.clear_mask_features()

    def training_epoch_end(self, outputs) -> None:
        if self.args.ft_with_knn:
            self.clear_cls_features()
        else:
            self.clear_mask_features()

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, _ = batch
        if self.args.ft_with_knn:
            output, cls_features = self.model(input_ids, attention_mask, return_dict=True)
            logits = output.logits
            logits, knn_logits, combine_logits = self.pvp(logits, input_ids, mode='val', cls_features=cls_features)
        else:
            output, lm_features = self.model(input_ids, attention_mask, return_dict=True)
            logits = output.logits
            logits, knn_logits, combine_logits = self.pvp(logits, input_ids, mode='val', lm_features=lm_features)
        
        if self.args.train_with_knn:
            loss_fct = knnLoss()
            loss = loss_fct(logits, knn_logits, labels, self.args.alpha)
        else:
            loss = self.loss_fn(logits, labels)
        if combine_logits is not None:
            logits = combine_logits
        self.log("Eval/loss", loss)
        return {"eval_logits": logits.detach().cpu().numpy(), "eval_labels": labels.detach().cpu().numpy()}
    
    def on_validation_epoch_start(self):
        if self.args.train_with_knn or self.args.knn_infer:
            train_dataloader = self.datamodule.train_dataloader()
            if self.args.ft_with_knn:
                self.get_cls_features_for_knn(train_dataloader)
            else:
                self.get_mask_features_for_knn(train_dataloader)
            
    def on_validation_epoch_end(self):
        if self.args.ft_with_knn:
            self.clear_cls_features()
        else:
            self.clear_mask_features()

        if self.args.ssl:
            ssl_dataloader = self.datamodule.train_dataloader(shuffle=False)
            pseudo_labels = self.get_ssl_pseudo_labels(ssl_dataloader)
            pseudo_dataset = self.set_ssl_pseudo_labels(self.datamodule.data_train, pseudo_labels)
            self.datamodule.data_train = pseudo_dataset

    
    def validation_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["eval_logits"] for o in outputs])
        labels = np.concatenate([o["eval_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Eval/f1", f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
        self.log("Eval/best_f1", self.best_f1, prog_bar=True, on_epoch=True)
        if self.args.ft_with_knn:
            self.clear_cls_features()
        else:
            self.clear_mask_features()

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        input_ids, attention_mask, labels, _ = batch
        if self.args.ft_with_knn:
            output, cls_features = self.model(input_ids, attention_mask, return_dict=True)
            logits = output.logits
            logits, knn_logits, combine_logits = self.pvp(logits, input_ids, mode='test', cls_features=cls_features)
        else:
            output, lm_features = self.model(input_ids, attention_mask, return_dict=True)
            logits = output.logits
            logits, knn_logits, combine_logits = self.pvp(logits, input_ids, mode='test', lm_features=lm_features)
        
        if combine_logits is not None:
            logits = combine_logits

        if self.args.knn_only:
            logits = knn_logits

        return {"test_logits": logits.detach().cpu().numpy(), "test_labels": labels.detach().cpu().numpy()}

    def on_test_epoch_start(self):
        if self.args.train_with_knn or self.args.knn_infer:
            train_dataloader = self.datamodule.train_dataloader()
            if self.args.ft_with_knn:
                self.get_cls_features_for_knn(train_dataloader)
            else:
                self.get_mask_features_for_knn(train_dataloader)
            
    def on_test_epoch_end(self):
        if self.args.ft_with_knn:
            self.clear_cls_features()
        else:
            self.clear_mask_features()
        
    def test_epoch_end(self, outputs) -> None:
        logits = np.concatenate([o["test_logits"] for o in outputs])
        labels = np.concatenate([o["test_labels"] for o in outputs])

        f1 = self.eval_fn(logits, labels)['f1']
        self.log("Test/f1", f1)    

    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument("--t_lambda", type=float, default=0.01, help="")
        parser.add_argument("--t_gamma", type=float, default=0.3, help="")
        return parser
        
    def pvp(self, logits, input_ids, mode, is_ssl=False, lm_features=None, cls_features=None):
        # convert the [batch_size, seq_len, vocab_size] => [batch_size, num_labels]
        #! hard coded
        if lm_features is not None:
            _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            bs = input_ids.shape[0]
            mask_output = logits[torch.arange(bs), mask_idx]
            lm_features = lm_features[torch.arange(bs), mask_idx]
            assert mask_idx.shape[0] == bs, "only one mask in sequence!"
            logits = mask_output[:,self.word2label]
        knn_logits, combine_logits = None, None
        
        if self.args.knn_infer and (self.args.train_with_knn or mode == 'test') and not is_ssl:
            # for knn
            bsz = input_ids.size(0)
            if self.args.ft_with_knn:
                cls_embedding = np.array(cls_features.cpu().detach(), dtype=np.float32)
                topk = self.args.knn_topk
                D, I = self.cls_features.search(cls_embedding, topk)
                D = torch.from_numpy(D).to(input_ids.device)
                # filter no_relation
                for i in range(bsz):
                    for j in range(topk):
                        try:
                            if self.clsid2labelid[I[i][j]] == self.rel2id['Other']:
                                D[i][j] = -1000
                        except:
                            if self.clsid2labelid[I[i][j]] == self.rel2id['no_relation']:
                                D[i][j] = -1000
                knn_logits = torch.full((bsz, self.num_labels), 0.).to(input_ids.device)
                for  i in range(bsz):
                    '''like knnlm'''
                    soft_knn_i = torch.softmax(-D[i]/self.args.temp, dim=-1)
                    for j in range(topk):
                        knn_logits[i][self.clsid2labelid[I[i][j]]] += soft_knn_i[j]
                
                cls_logits = torch.softmax(logits, dim=-1)
                combine_logits = combine_knn_and_vocab_probs(knn_logits, cls_logits, coeff=self.args.knn_lambda)
            else:
                mask_embedding = np.array(lm_features.cpu().detach(), dtype=np.float32)
                topk = self.args.knn_topk
                D, I = self.mask_features.search(mask_embedding, topk)
                D = torch.from_numpy(D).to(input_ids.device)
                # filter no_relation
                for i in range(bsz):
                    for j in range(topk):
                        try:
                            if self.maskid2labelid[I[i][j]] == self.rel2id['Other']:
                                D[i][j] = -1000
                        except:
                            if self.maskid2labelid[I[i][j]] == self.rel2id['no_relation']:
                                D[i][j] = -1000
                knn_logits = torch.full((bsz, self.num_labels), 0.).to(input_ids.device)
                for i in range(bsz):
                    '''like knnlm'''
                    soft_knn_i = torch.softmax(-D[i]/self.args.temp, dim=-1)   # 1 x topk
                    for j in range(topk):
                        knn_logits[i][self.maskid2labelid[I[i][j]]] += soft_knn_i[j]
                
                mask_logits = torch.softmax(logits, dim=-1)
                combine_logits = combine_knn_and_vocab_probs(knn_logits, mask_logits, coeff=self.args.knn_lambda)
    
        return logits, knn_logits, combine_logits

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        if not self.args.two_steps: 
            parameters = self.model.named_parameters()
        else:
            # model.bert.embeddings.weight
            parameters = [next(self.model.named_parameters())]
        # only optimize the embedding parameters
        optimizer_group_parameters = [
            {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in parameters if any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * 0.1, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }

    def get_mask_features_for_knn(self, dataloader):
        self.eval()
        with torch.no_grad():
            for inputs in tqdm(dataloader, desc='KNN'):
                inputs = tuple(input.to(self.device) for input in inputs)
                self.training_step(inputs, batch_idx=None, save_mask=True)
            self.total_features = np.concatenate(self.total_features, axis=0)
            self.mask_features.add(self.total_features)
        self.train()

    def get_cls_features_for_knn(self, dataloader):
        self.eval()
        with torch.no_grad():
            for inputs in tqdm(dataloader, desc='KNN'):
                inputs = tuple(input.to(self.device) for input in inputs)
                self.training_step(inputs, batch_idx=None, save_cls=True)
            self.total_features = np.concatenate(self.total_features, axis=0)
            self.cls_features.add(self.total_features)
        self.train()

    def clear_mask_features(self):
        self.total_features = []
        self.mask_features = faiss.IndexFlatL2(self.model.config.hidden_size)
        self.maskid2labelid = {}
        self.cnt_batch = 0

    def clear_cls_features(self):
        self.total_features = []
        self.cls_features = faiss.IndexFlatL2(self.model.config.hidden_size)
        self.clsid2labelid = {}
        self.cnt_batch = 0

    def get_ssl_pseudo_labels(self, dataloader):
        pseudo_labels = []
        with torch.no_grad():
            for inputs in tqdm(dataloader, desc='SSL'):
                inputs = tuple(input.to(self.device) for input in inputs)
                output = self.training_step(inputs, batch_idx=None, is_ssl=True)
                logits = output['logits']
                preds = torch.argmax(logits, dim=-1)
                pseudo_labels.extend(preds.detach().cpu().tolist())
        return pseudo_labels

    def set_ssl_pseudo_labels(self, dataset, pesudo_labels):
        input_ids, attention_mask, labels, so = dataset.tensors
        pesudo_labels = torch.tensor(pesudo_labels)
        assert pesudo_labels.size() == labels.size()
        dataset = TensorDataset(input_ids, attention_mask, pesudo_labels, so)
        return dataset



def decode(tokenizer, output_ids):
    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output_ids]


def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff=0.5):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

    return curr_prob
