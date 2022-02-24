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
from transformers import LongformerTokenizerFast

class WIKIPEDIAVAL(data.Dataset):

    def __init__(
            self, 
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        conf = DownloadConfig(cache_dir='/data5/nguyenjd/wikipedia1')
        self.dataset = load_dataset("wikipedia", "20200501.en", split="train[90%:]",cache_dir='/data5/nguyenjd/wikipedia1', download_config=conf)
        #self.tokenizer = t.RobertaTokenizerFast.from_pretrained('roberta-base')
        #breakpoint()
        self.tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096') #might not need from_pretrained
        #self.tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096') #might not need from_pretrained
        self.train = train
        self.span_length = 32 

    def getRandomSubstring(self, tok_ids):
        """ 
        length = len(tok_ids)
        if(length <= 2 * self.span_length):
            return torch.Tensor(tok_ids[0:self.span_length]), 0
        else: 
            ind = random.randint(0, length-(2*self.span_length) - 1) #2 because of the neighboring span
        return torch.Tensor(tok_ids[ind:ind+self.span_length]), ind
        """
        length = len(tok_ids)

        
        if length <= self.span_length-2:
            ret_str = tok_ids
            ind = 0
        else:
            ind = random.randint(0, length-self.span_length+2)
            ret_str = tok_ids[ind:]
        
        substr_tok = self.tokenizer.encode(self.tokenizer.decode(ret_str), padding='max_length', truncation=True, max_length=self.span_length, pad_to_max_length=True)
        return torch.Tensor(substr_tok), ind
        
    def getNeighborSubstring(self, tok_ids, ind):
        
        length = len(tok_ids)
        if(length <= 2 * self.span_length):
            substr_tok_tensor, idx = self.getRandomSubstring(tok_ids)
            return substr_tok_tensor
        else:
            ret_str = tok_ids[ind+self.span_length-2:]
        substr_tok = self.tokenizer.encode(self.tokenizer.decode(ret_str), padding='max_length', truncation=True, max_length=self.span_length, pad_to_max_length=True)
        return torch.Tensor(substr_tok)
        
        
    def getRandomIndex(self, index):
        #num = index
        #while num == index:
        num = random.randint(0, len(self.dataset)-1)
        return num

    def getNotNeighborSubstring(self, data, index):
        """
        text = data['text']
        split = text.split()
        length = len(split)
        num = index
        #while num > index - self.span_length and num < index + self.span_length:
        num = random.randint(0, length-1)
        
        substr = split[num:]
        strsent = " ".join(str(x) for x in substr)
        return strsent       
        """
        pass
    def __getitem__(self, index):
        
        data1 = self.dataset.__getitem__(index)        
        random = self.getRandomIndex(index)
        data2 = self.dataset.__getitem__(random)        
        
        tok_ids1 = self.tokenizer.encode(data1['text'], add_special_tokens=False)
        tok_ids2 = self.tokenizer.encode(data2['text'], add_special_tokens=False)

        orig_tok_tensor, orig_idx = self.getRandomSubstring(tok_ids1)
        same_doc_tensor, same_doc_idx = self.getRandomSubstring(tok_ids1)
        diff_doc_tensor, diff_doc_idx = self.getRandomSubstring(tok_ids2)

        neighbor_tok_tensor = self.getNeighborSubstring(tok_ids1, orig_idx)
        not_neighbor_tok_tensor, not_neighbor_idx = self.getRandomSubstring(tok_ids1)
        

        return index, orig_tok_tensor.long(), same_doc_tensor.long(), diff_doc_tensor.long(), neighbor_tok_tensor.long(), not_neighbor_tok_tensor.long()
        
    def __len__(self):
        #return len(self.dataset)
        return 1000
