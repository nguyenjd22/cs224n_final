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
import transformers as t
from sklearn.datasets import fetch_20newsgroups


class NEWSGROUP(data.Dataset):

    def __init__(
            self, 
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        #self.dataset = load_dataset("newsgroup")
        if train:
            self.newsgroup_list = fetch_20newsgroups(subset='train')
        else:
            self.newsgroup_list = fetch_20newsgroups(subset='test')
        
        self.dataset = self.newsgroup_list.data
        
        self.tokenizer = t.LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
        #self.tokenizer = t.BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.train = train
        self.span_length = 32


    def getRandomSubstring(self, tok_ids):
       
        length = len(tok_ids)

        if length <= self.span_length-2:
            ret_str = tok_ids
        else:
            ind = random.randint(0, length-self.span_length+2)
            ret_str = tok_ids[ind:]
        
        substr_tok = self.tokenizer.encode(self.tokenizer.decode(ret_str), padding='max_length', truncation=True, max_length=self.span_length, pad_to_max_length=True)
        return torch.Tensor(substr_tok)
          
    def __getitem__(self, index):

        data = self.dataset[index]
        
        tok_ids = self.tokenizer.encode(data, add_special_tokens=False)

        tok_ids_tensor = self.getRandomSubstring(tok_ids)
        
        label = self.newsgroup_list.target[index] 
         
        return index, tok_ids_tensor.long(), label

    def __len__(self):
        return len(self.dataset)

