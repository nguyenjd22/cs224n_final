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


class WIKIPEDIA(data.Dataset):

    def __init__(
            self, 
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        #breakpoint()
        conf = DownloadConfig(cache_dir='/data5/nguyenjd/wikipedia1')
        self.dataset = load_dataset("wikipedia", "20200501.en", split='train[:90%]', cache_dir='/data5/nguyenjd/wikipedia1', download_config=conf)
        #self.tokenizer = t.LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
        self.tokenizer = t.LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
        #self.tokenizer = t.RobertaTokenizerFast.from_pretrained('roberta-base')
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
        data = self.dataset.__getitem__(index)
        if self.train == True: 
            span = self.getRandomSubstring(data)
        else: 
            span = data['text']
        tok_ids = self.tokenizer.encode(span, truncation=True, padding='max_length', max_length=self.span_length, pad_to_max_length=True)
        tok_ids_tensor = torch.Tensor(tok_ids) 
        return index, tok_ids_tensor.long()

    def __len__(self):
        return len(self.dataset)

class WIKIPEDIATwoViews(WIKIPEDIA):
    
    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)        

        #tok_ids = self.tokenizer.encode(data['text'], padding='max_length', max_length=self.span_length, pad_to_max_length=True)
        
        tok_ids = self.tokenizer.encode(data['text'], add_special_tokens=False)

        tok_ids_tensor1 = self.getRandomSubstring(tok_ids)
        tok_ids_tensor2 = self.getRandomSubstring(tok_ids) 
        
        return index, tok_ids_tensor1.long(), tok_ids_tensor2.long()
