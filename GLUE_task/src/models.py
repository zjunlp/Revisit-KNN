"""Custom models for few-shot learning specific operations."""

import torch
import torch.nn as nn
import numpy as np
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from .modeling_roberta import RobertaClassificationHead, RobertaLMHead, RobertaModel
from .utils import knnLoss
import faiss
import logging
logger = logging.getLogger(__name__)


class RobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None
        
        # For saving [MASK]
        self.cnt_batch = 0  # record current batch
        self.maskid2labelid = {}
        d = config.hidden_size
        if config.sim_metric == 'L2':
            self.mask_features = faiss.IndexFlatL2(d)
        elif config.sim_metric == 'cosine':
            self.mask_features = faiss.IndexFlatIP(d)   # cosine
        self.total_features = []
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        return_output=False,
        save_mask=False,
        is_ssl=False
    ):
        bsz = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores, lm_feature = self.lm_head(sequence_mask_output)

        # just for knn save features
        if save_mask:
            self.total_features.append(lm_feature.cpu().detach())
            for idx, label in zip(range(bsz), labels):
                self.maskid2labelid[idx + self.cnt_batch] = label.cpu().detach()
            self.cnt_batch += bsz
            return None
            
        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        knn_logits, combine_logits = None, None
        logits = prediction_mask_scores[:, self.label_word_list] 
        if len(logits.shape) == 3:  # multi-token label:(4, 14, 2) bsz, n_label, n_per_label
            logits = logits.sum(-1)
            
        # ssl return logits
        if is_ssl:
            return logits
            
        # for knn inference
        if self.model_args.knn_infer:
            bsz = input_ids.size(0)
            mask_embedding = np.array(lm_feature.cpu().detach(), dtype=np.float32)
            # norm mask embedding for consine similartiy
            faiss.normalize_L2(mask_embedding)
            topk = self.model_args.knn_topk
            D, I = self.mask_features.search(mask_embedding, topk)
            D = torch.from_numpy(D).to(input_ids.device)
            knn_logits = torch.full((bsz, self.num_labels), 0.).to(input_ids.device)
            for i in range(bsz):
                ''''like knnlm'''
                if self.model_args.sim_metric == 'L2':
                    soft_knn_i = torch.softmax(-D[i]/self.model_args.temp, dim=-1)   # 1 x topk L2 distance
                elif self.model_args.sim_metric == 'consine':
                    soft_knn_i = torch.softmax(D[i]/self.model_args.temp, dim=-1)   # 1 x topk consine
                for j in range(topk):
                    knn_logits[i][self.maskid2labelid[I[i][j]]] += soft_knn_i[j]
            
            if self.model_args.only_knn_infer and self.model_args.ssl:
                logits = knn_logits
            else:
                mask_logits = torch.softmax(logits, dim=-1)
                combine_logits = combine_knn_and_vocab_probs(knn_logits, mask_logits, coeff=self.model_args.knn_lambda)

        loss = None
        if labels is not None:
            # knn for train
            if self.model_args.train_with_knn and knn_logits is not None: 
                loss_fct = knnLoss()
                loss = loss_fct(logits, knn_logits, labels.view(-1), self.model_args.alpha)
            else:
                loss_fct = nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        if combine_logits is not None:
            logits = combine_logits

        if return_output:
            outputs = (logits, sequence_mask_output)
        else:
            outputs = (logits, )
    
        return ((loss,) + outputs) if loss is not None else outputs


def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff=0.5):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = np.log(1 - coeff)
    coeffs[1] = np.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

    return curr_prob


class RobertaForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()
        
        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None
        
        # For saving [MASK]
        self.cnt_batch = 0  # record current batch
        self.maskid2labelid = {}
        d, measure = config.hidden_size, faiss.METRIC_L2
        self.mask_features = faiss.IndexFlatL2(d)
        self.total_features = []

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_output=None,
        save_mask=False,
        is_ssl=False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        bsz = input_ids.size(0)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        cls_features = sequence_output[:, 0, :]
        logits = self.classifier(sequence_output)
        
        # ssl return logits
        if is_ssl:
            return logits
        
        # TODO: save cls features
        if save_mask:
            self.total_features.append(cls_features.cpu().detach())
            for idx, label in zip(range(bsz), labels):
                self.maskid2labelid[idx + self.cnt_batch] = label.cpu().detach()
            self.cnt_batch += bsz
            return None

        combine_logits = None
        # for knn inference
        if self.model_args.knn_infer:
            bsz = input_ids.size(0)
            cls_embedding = np.array(cls_features.cpu().detach(), dtype=np.float32)
            topk = self.model_args.knn_topk
            D, I = self.mask_features.search(cls_embedding, topk)
            D = torch.from_numpy(D).to(input_ids.device)
            knn_logits = torch.full((bsz, self.num_labels), 0.).to(input_ids.device)
            for i in range(bsz):
                ''''like knnlm'''
                soft_knn_i = torch.softmax(-D[i]/self.model_args.temp, dim=-1)   # 1 x topk
                for j in range(topk):
                    knn_logits[i][self.maskid2labelid[I[i][j]]] += soft_knn_i[j]
            
            if self.model_args.only_knn_infer and self.model_args.ssl:
                logits = knn_logits
            else:
                mask_logits = torch.softmax(logits, dim=-1)
                combine_logits = combine_knn_and_vocab_probs(knn_logits, mask_logits, coeff=self.model_args.knn_lambda)
           
        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            # knn for train
            if self.model_args.knn_infer:
                loss_fct = knnLoss()
                loss = loss_fct(logits, knn_logits, labels.view(-1), self.model_args.alpha)
            else:
                loss_fct = nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        if combine_logits is not None:
            logits = combine_logits

        if return_output:
            outputs = (logits, sequence_output)
        else:
            outputs = (logits, )
    
        return ((loss,) + outputs) if loss is not None else outputs
