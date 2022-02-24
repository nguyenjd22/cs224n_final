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


class WIKIPEDIASLICE(data.Dataset):

    def __init__(
            self, 
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        conf = DownloadConfig(cache_dir='/data5/nguyenjd/wikipedia1')
        self.dataset = load_dataset("wikipedia", "20200501.en", split='train[:90%]', cache_dir='/data5/nguyenjd/wikipedia1', download_config=conf)
        self.tokenizer = t.LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
        #self.tokenizer = t.RobertaTokenizerFast.from_pretrained('roberta-base')
        self.train = train
        self.span_length = 32
        self.mask_len = int(self.span_length / (4*2)) #25% are masked with using 2 masks
        
    def getRandomMaskSubstring(self, tok_ids, ind, mask_pos, even):
        
        length = len(tok_ids)
        if length <= self.span_length-2:
            ret_str = tok_ids
        else:
            ret_str = tok_ids[ind:ind+self.span_length+6]
        
                #first_num = random.randint(0, len(ret_str) - (2 * mask_len) - 1) #leaves 1 space in between masks
        #second_num = random.randint(first_num + mask_len + 1, len(ret_str) - mask_len)
        if even:
            first_num = min(mask_pos[0], mask_pos[1]) * self.mask_len
            second_num = max(mask_pos[0], mask_pos[1]) * self.mask_len
        else: 
            first_num = min(mask_pos[0], mask_pos[1]) * self.mask_len + 1
            second_num = max(mask_pos[0], mask_pos[1]) * self.mask_len + 1
 

        ret_str[first_num] = self.tokenizer.mask_token_id
        ret_str[second_num] = self.tokenizer.mask_token_id
        
        del ret_str[second_num+1:second_num + self.mask_len]
        del ret_str[first_num+1: first_num + self.mask_len]

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

class WIKIPEDIASLICETwoViews(WIKIPEDIASLICE):
    
    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)        

        tok_ids = self.tokenizer.encode(data['text'], add_special_tokens=False)
        length = len(tok_ids)
        if length <= 32:
            substr_tok = self.tokenizer.encode(self.tokenizer.decode(tok_ids), padding='max_length', truncation=True, max_length=self.span_length, pad_to_max_length=True)
            return index, torch.Tensor(substr_tok).long(), torch.Tensor(substr_tok).long() 

        ind = random.randint(0, length-self.span_length)
        evens = random.sample(range(0, int(self.span_length / (self.mask_len * 2))), 2)
        odds = random.sample(range(0, int(self.span_length / (self.mask_len * 2))), 2)
        
        tok_ids_tensor1 = self.getRandomMaskSubstring(tok_ids, ind, evens, even=True)
        tok_ids_tensor2 = self.getRandomMaskSubstring(tok_ids, ind, odds, even=False) 
        
        return index, tok_ids_tensor1.long(), tok_ids_tensor2.long()
