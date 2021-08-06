import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_metric
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoConfig
import numpy as np

# pos_tags: a list of classification labels, with possible values including " (0), '' (1), # (2), $ (3), ( (4).
# chunk_tags: a list of classification labels, with possible values including O (0), B-ADJP (1), I-ADJP (2), B-ADVP (3), I-ADVP (4).
# ner_tags: a list of classification labels, with possible values including O (0), B-PER (1), I-PER (2), B-ORG (3), I-ORG (4) B-LOC (5), I-LOC (6) B-MISC (7), I-MISC (8).

POS_LIST = [ '"', "''", '#', '$', '(']
CHUNK_LIST = ['O', 'B-ADJP', 'I-ADJP', 'B-ADVP', 'I-ADVP']
NER_LIST = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

MAP_DICT = {
    'ner': NER_LIST,
    'chunk': CHUNK_LIST,
    'pos': POS_LIST
}

class CoNLL(Dataset):
    def __init__(self, task: str, data: HFDataset, model_name:str) -> None:
        super().__init__()
        self.task = task + '_tags'
        self.input, self.labels, self.label_mask, self.attention_mask = [], [], [], []
        self.ignore_columns = ['ner_tags','pos_tags', 'id', 'chunk_tags', 'tokens']
        
        aps = True if 'berta' in model_name else False
        tokenizer_name = 'microsoft/deberta-xlarge' if 'deberta' in model_name else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            revision='main',
            add_prefix_space=aps,
        )

        # self.label_list = self.get_label_list(data['train'][self.task])
        self.label_list = MAP_DICT[task]
        self.label_to_id = {l: i for i, l in enumerate(self.label_list)}
        num_labels = len(self.label_list)

        self.train_data = data['train'].map(
            self.tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
        )
        self.dev_data = data['validation'].map(
            self.tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
        )
        self.test_data = data['test'].map(
            self.tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
        )

        self.data_collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8)
        self.metric = load_metric('data/metric.py')
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=self.label_to_id,
            id2label={i: l for l, i in self.label_to_id.items()},
            finetuning_task=task,
            revision='main',
        )

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples['tokens'],
            padding=False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        # print(tokenized_inputs.keys())
        # exit()
        labels = []
        for i, label in enumerate(examples[self.task]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def get_label_list(self, labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
