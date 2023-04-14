import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import torch

import numpy as np

from transformers import RobertaTokenizer, AutoConfig, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed

from src.dataset import FewShotDataset
from src.models import RobertaForPromptFinetuning, RobertaForSequenceClassification
from src.trainer import Trainer
from src.processors import  num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping

from datetime import datetime


logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    few_shot_type: str = field(
        default='prompt',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt"}
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )
    # KNN hyper parameters
    knn_lambda: float = field(
        default=0.2, 
    )
    knn_topk: Optional[int] = field(
        default=32,
        metadata={"help": "knn topk"}
    )
    temp: float = field(
        default=5, 
    )
    alpha: float = field(
        default=0.001, 
    )


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )
    # For prompting
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )


    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: list = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."}
    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )
    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )
    evaluate_during_training: bool = field(
        default=True,
        metadata={"help": "evaluate during training."}
    )
    ckpt_dir: str = field(
        default=None
    )
    train_with_loss: bool = field(
        default=False
    )
    # knn
    train_with_knn: bool = field(
        default=False, 
        metadata={"help": "UNION-all, kNN as Prior Knowledge for Calibrating Training."}
    )
    only_train_knn: bool = field(
        default=False, 
        metadata={"help": "only kNN as Prior Knowledge for Calibrating Training."}
    )
    knn_infer: bool = field(
        default=False, 
        metadata={"help": "Integrating kNN into Inference."}
    )
    ssl: bool = field(
        default=False, 
        metadata={"help": "Zero-shot learning."}
    )
    only_knn_infer: bool = field(
        default=False, 
        metadata={"help": "UNION-infer, only integrating kNN into Inference."}
    )
    sim_metric: str = field(
        default="L2", 
        metadata={"help": "Similartiy metrics. Choice: L2, cosine."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    data_args.output_dir = training_args.output_dir
    model_args.train_with_knn = training_args.train_with_knn
    model_args.knn_infer = training_args.knn_infer
    model_args.only_knn_infer = training_args.only_knn_infer
    model_args.ssl = training_args.ssl
    model_args.sim_metric = training_args.sim_metric
    
    print(model_args)
    print(data_args)
    print(training_args)
    
    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))
    
    # Create config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name
    )

    if 'prompt' in model_args.few_shot_type:
        if config.model_type == 'roberta':
            model_fn = RobertaForPromptFinetuning
        else:
            raise NotImplementedError
    elif model_args.few_shot_type == 'finetune':
        model_fn = RobertaForSequenceClassification
    else:
        raise NotImplementedError
    special_tokens = []

    # Create tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
    )

    # Get our special datasets.
    train_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="train")
    )
    eval_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="dev")
        if training_args.do_eval
        else None
    )
    test_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="test")
        if training_args.do_predict
        else None
    )

    config.sim_metric = model_args.sim_metric
    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )


    # Pass dataset and argument information to the model
    if data_args.prompt:
        model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
    if output_modes_mapping[data_args.task_name] == 'regression':
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
        
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            logits = p.predictions
            preds = np.argmax(logits, axis=1)
            label_ids = p.label_ids

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)
        
        return compute_metrics_fn
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name)
    )

    # Training
    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
        # Use the early stop, so do not save the model in the end (unless specify save_at_last)
        tokenizer.save_pretrained(training_args.output_dir)
        torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
        torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))
        
        # Reload the best checkpoint (for eval)
        model = model_fn.from_pretrained(training_args.output_dir)
        model = model.to(training_args.device)
        trainer.model = model
        if data_args.prompt:
            model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
        if output_modes_mapping[data_args.task_name] == 'regression':
            # lower / upper bounds
            model.lb, model.ub = bound_mapping[data_args.task_name]
        model.model_args = model_args
        model.data_args = data_args
        model.tokenizer = tokenizer

    # Evaluation
    final_result = {
        'time': str(datetime.today()),
    }
    if not training_args.do_train and not (training_args.only_knn_infer and training_args.ssl):
        if training_args.ckpt_dir is not None:
            model = model_fn.from_pretrained(training_args.ckpt_dir)
        else:
            model = model_fn.from_pretrained(training_args.output_dir)
        model = model.to(training_args.device)
        trainer.model = model
        if data_args.prompt:
            model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
        if output_modes_mapping[data_args.task_name] == 'regression':
            # lower / upper bounds
            model.lb, model.ub = bound_mapping[data_args.task_name]
        model.model_args = model_args
        model.data_args = data_args
        model.tokenizer = tokenizer

    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics 

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if True:
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[eval_dataset.args.task_name + '_dev_' + key] = value
            eval_results.update(eval_result)

    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                for key, value in test_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
                    final_result[test_dataset.args.task_name + '_test_' + key] = value

            test_results.update(test_result)
    
    return eval_results

if __name__ == "__main__":
    main()
