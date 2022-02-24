import os
import random
import copy
import getpass
from PIL import Image
import numpy as np
from nlp import load_dataset
from nlp import DownloadConfig
import torch
import torch.utils.data as data
from torchvision import transforms, datasets
import transformers

SPAN_LENGTH = 32  # TODO: increase

class GlueDataset(data.Dataset):
    def __init__(
        self,
        task_name,
        train=True,
        span_len=SPAN_LENGTH,
    ):
        super().__init__()
        self.task_name = task_name
        if task_name == 'mnli':
            split = 'train' if train else 'validation_mismatched'
        else:
            split = 'train' if train else 'validation'
        self.dataset = self.get_dataset(task_name, split)
        self.tokenizer = transformers.LongformerTokenizerFast.from_pretrained(
            'allenai/longformer-base-4096')
        self.tokenizer.add_prefix_space = True
        self.train = train
        self.span_len = span_len
    def get_dataset(self, task_name, split):
        return load_dataset('glue', task_name, split=split)
    def __getitem__(self, index):
        ex = self.dataset[index]
        if self.task_name in ['cola', 'sst2']:
            txt_input = (ex['sentence'],)
        elif self.task_name in ['qqp']:
            txt_input = (ex['question1'], ex['question2'])
        elif self.task_name in ['mrpc', 'rte', 'wnli']:
            txt_input = (ex['sentence1'], ex['sentence2'])
        elif self.task_name in ['mnli']:
            txt_input = (ex['premise'], ex['hypothesis'])
        elif self.task_name in ['qnli']:
            txt_input = (ex['question'], ex['sentence'])
        else:
            raise ValueError(f'Task {self.task_name} not defined')
        input_ids = self.tokenizer.encode(*txt_input, truncation=True, padding='max_length', max_length=self.span_len,
                                          pad_to_max_length=True, return_tensors='pt')[0]
        label_id = self.dataset[index]['label']
        return index, input_ids, label_id
    def __len__(self):
        return len(self.dataset)
