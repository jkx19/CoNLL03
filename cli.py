import datasets
from datasets.load import load_metric, load_dataset, load_dataset_builder
import numpy as np
import torch
from torch import Tensor
import torch.nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.optim import AdamW
from transformers.trainer_pt_utils import get_parameter_names

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
import sys
import os
import argparse

from data.conll_dataset import CoNLL
from model.prefix import BertForTokenClassification, BertPrefixModel
from model.prefix import DeBertaPrefixModel
from model.prefix import DeBertaV2PrefixModel
from model.deberta import DebertaForTokenClassification
from trainer import Trainer

ADD_PREFIX_SPACE = {
    'bert': False,
    'deberta': True,
    'gpt2': True,
    'debertaV2': True,
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )


# METRIC: F1 score
# Note: the main reason abandoning LAMA is to fit the metric

class Trainer_API:    
    def __init__(self, args) -> None:

        # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))        
        # self.model_args, self.data_args, self.training_args = parser.parse_args_into_dataclasses()

        self.task = args.task
        assert self.task in ['pos', 'chunk', 'ner']
        self.device = torch.device('cuda:0')
        
        device_num = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.batch_size = args.batch_size * device_num
        self.epoch = args.epoch
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.weight_decay = 0
        self.gamma = 0.95
        self.lr = args.lr

        if args.model == 'bert':
            self.model_name = f'bert-{args.model_size}-uncased'
        elif args.model == 'deberta':
            if args.model_size == 'base':
                self.model_name = 'microsoft/deberta-xlarge'
            elif args.model_size == 'large':
                raise NotImplementedError
        elif args.model == 'debertaV2':
            if args.model_size == 'base':
                self.model_name = 'microsoft/deberta-xlarge-v2'
            elif args.model_size == 'large':
                self.model_name = 'microsoft/deberta-xxlarge-v2'
        elif args.model == 'gpt2':
            if args.model_size == 'base':
                self.model_name = 'gpt2-medium'
            elif args.model_size == 'large':
                self.model_name = 'gpt2-large'
        
        raw_data = load_dataset('data/load_dataset.py')
        add_prefix_space = ADD_PREFIX_SPACE[args.model]
        dataset = CoNLL(self.task, raw_data, self.model_name, aps=add_prefix_space)

        self.train_dataset = dataset.train_data
        self.dev_dataset = dataset.dev_data
        self.test_dataset = dataset.test_data
        self.ignore_columns = dataset.ignore_columns

        self.tokenizer = dataset.tokenizer
        self.data_collator = dataset.data_collator
        self.compute_metrics = dataset.compute_metrics
        self.lm_config = dataset.config

        self.method = args.method
        if args.method == 'prefix':
            self.lm_config.hidden_dropout_prob = args.dropout
            self.lm_config.pre_seq_len = args.pre_seq_len
            self.lm_config.mid_dim = args.mid_dim
            if args.model == 'deberta':
                self.model = DeBertaPrefixModel.from_pretrained(
                    self.model_name,
                    config=self.lm_config,
                    revision='main',
                )
            elif args.model == 'debertaV2':
                self.model = DeBertaV2PrefixModel.from_pretrained(
                    self.model_name,
                    config=self.lm_config,
                    revision='main',
                )
            elif args.model == 'bert':
                self.model = BertPrefixModel.from_pretrained(
                    self.model_name,
                    config=self.lm_config,
                    revision='main',
                )
            elif args.model == 'gpt2':
                raise NotImplementedError

        elif args.method == 'finetune':
            if 'deberta' in self.model_name:
                self.model = DebertaForTokenClassification.from_pretrained(
                    self.model_name,
                    config=self.lm_config,
                    revision='main',
                )
            elif 'bert' in self.model_name:
                self.model = BertForTokenClassification.from_pretrained(
                    self.model_name,
                    config=self.lm_config,
                    revision='main',
                )


        self.train_loader = self.get_data_loader(self.train_dataset)
        self.dev_loader = self.get_data_loader(self.dev_dataset)
        self.test_loader = self.get_data_loader(self.test_dataset)
        max_dev_len = max([batch['labels'].shape[1] for _, batch in enumerate(self.dev_loader)])
        max_test_len = max([batch['labels'].shape[1] for _, batch in enumerate(self.test_loader)])
        self.max_seq_len = max(max_dev_len, max_test_len)

    def get_sampler(self, dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        # Build the sampler.
        return RandomSampler(dataset, generator=generator)

    def get_data_loader(self, dataset: datasets.arrow_dataset.Dataset) -> DataLoader:
        dataset = dataset.remove_columns(self.ignore_columns)
        sampler = self.get_sampler(dataset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )

    def get_optimizer(self):
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.classifier.named_parameters() if n in decay_parameters],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.classifier.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.model.prefix_encoder.named_parameters() if n not in decay_parameters],
                "weight_decay": self.weight_decay,
            },
        ]            
        optimizer_kwargs = {
            "betas": (self.adam_beta1, self.adam_beta2),
            "eps": self.adam_epsilon,
        }
        optimizer_kwargs["lr"] = self.lr            
        self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

    def get_schedular(self):
        pass

    def pad_tensor(self, tensor: torch.Tensor, pad_index: int):
        r'''
        Pad the ( batched ) result tensor to max length for concatent with given pad-index
        '''
        max_size = self.max_seq_len
        old_size = tensor.shape
        new_size = list(old_size)
        new_size[1] = max_size
        new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
        new_tensor[:, : old_size[1]] = tensor
        return new_tensor

    def train(self):        
        self.get_optimizer()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.gamma)
        pbar = tqdm(total=len(self.train_loader)*self.epoch)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        best_dev_result = 0
        best_test_result = 0
        for epoch in range(self.epoch):
            # Train
            total_loss = 0
            self.model.train()
            for batch_idx, batch in enumerate(self.train_loader):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                output = self.model(**batch)
                loss = torch.sum(output.loss)
                # loss = output.loss
                total_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                pbar.update(1)
            self.scheduler.step()

            # Evaluate
            dev_result = self.eval()
            test_result = self.test()
            if best_dev_result < dev_result["f1"]:
                best_dev_result = dev_result["f1"]
                best_test_result = test_result
                best_head = self.model.classifier.state_dict()
            pbar.set_description(f'Train_loss: {total_loss:.1f}, Eval_F1: {dev_result["f1"]:.3f}, Test_F1: {test_result["f1"]:.3f},')

        pbar.close()
        return best_test_result
    
    def eval(self):
        self.model.eval()
        with torch.no_grad():
            labels, prediction = [], []
            for batch_idx, batch in enumerate(self.dev_loader):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                output = self.model(**batch)
                loss,logits = output.loss, output.logits
                logits = self.pad_tensor(logits, -100)
                prediction.append(logits)
                batch_label = self.pad_tensor(batch['labels'], -100)
                labels.append(batch_label)
            prediction = torch.cat(prediction)
            labels = torch.cat(labels)
            result = self.compute_metrics((np.array(prediction.cpu()), np.array(labels.cpu())))
        return result

    def test(self):
        self.model.eval()
        with torch.no_grad():
            labels, prediction = [], []
            for batch_idx, batch in enumerate(self.test_loader):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                output = self.model(**batch)
                loss,logits = output.loss, output.logits
                logits = self.pad_tensor(logits, -100)
                prediction.append(logits)
                batch_label = self.pad_tensor(batch['labels'], -100)
                labels.append(batch_label)
            prediction = torch.cat(prediction)
            labels = torch.cat(labels)
            result = self.compute_metrics((np.array(prediction.cpu()), np.array(labels.cpu())))
        return result


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--task', type=str, choices=['pos', 'chunk', 'ner'], default='ner')
    parser.add_argument('--pre_seq_len', type=int, default=4)
    parser.add_argument('--mid_dim', type=int, default=512)
    parser.add_argument('--model', type=str,choices=['bert', 'deberta', 'devertaV2'], default='deberta')
    parser.add_argument('--model_size', type=str, choices=['base', 'large'], default='base')
    parser.add_argument('--method', type=str, choices=['prefix', 'finetune'], default='prefix')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
    return args

def main():
    args = construct_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    train_api = Trainer_API(args)
    result = train_api.train()
    sys.stdout = open('result.txt', 'a')
    print(args)
    print(result)

if __name__ == '__main__':
    main()