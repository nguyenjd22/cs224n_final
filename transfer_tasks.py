import os, sys, logging
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaModel, LongformerConfig, LongformerTokenizerFast, LongformerModel
from transformers import BertTokenizerFast, BertModel
from src.utils.setup import process_config
import senteval 
import argparse
import torch
import numpy as np
import random 


parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, default='path to config file')
args = parser.parse_args()

config_path = args.config 
config = process_config(config_path)

transfer_tasks = ['SST2']


def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    tok_ids = [params.tokenizer.encode(s, add_special_tokens=False) for s in sentences]
    for i in range(len(tok_ids)):
        sent = tok_ids[i]
        length = len(sent)
        if len(sent) <= params.span_length-2:
            tok_ids[i] = params.tokenizer.encode(params.tokenizer.decode(sent), padding='max_length', max_length=params.span_length, pad_to_max_length=True, truncation=True)
        else:
            ind = random.randint(0, length-params.span_length + 2)
            rand_span = tok_ids[i][ind:]
            tok_ids[i] = params.tokenizer.encode(params.tokenizer.decode(rand_span), padding='max_length', max_length=params.span_length, pad_to_max_length=True, truncation=True)
    tok_ids = torch.Tensor(tok_ids).long()
    tok_ids = tok_ids.to('cuda:'+str(params.gpu_device))
    last_hidden_states = params.model(tok_ids)[0]
    average_hidden_states = last_hidden_states.mean(1) 
    return average_hidden_states.cpu().detach().numpy()

def prepare(params, samples):
    #RobertaModel with checkpoint 
    #configuration = RobertaConfig(hidden_size=config.model_params.out_dim, num_hidden_layers=2, num_attention_heads=2, intermediate_size=384, vocab_size=50265)
    params.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    #checkpoint = torch.load('/mnt/fs5/nguyenjd1/experiments/experiments/longformer4/checkpoints/epoch=2.ckpt')
    #params.model = RobertaModel.from_pretrained(pretrained_model_name_or_path=None, state_dict=checkpoint, config=configuration)
    params.model = BertModel.from_pretrained('bert-base-uncased')
    """ 
    #Longformer Pretrained
    configuration = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
#    configuration = LongformerConfig()
    configuration.num_attention_heads = 2
    configuration.intermediate_size = 384
    configuration.num_hidden_layers = 2
    configuration.attention_window = [4, 4]
    configuration.hidden_size = config.model_params.out_dim
    params.model = LongformerModel.from_pretrained(pretrained_model_name_or_path=None, state_dict=checkpoint, config=configuration)
    """


    
    #params.model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
    #params.tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
    params.gpu_device = config.gpu_device
    params.model.to('cuda:' + str(params.gpu_device))

params = {'task_path': '/home/nguyenjd/nguyenjd/SentEval/data', 'usepytorch': True, 'kfold': 1, 'span_length': 32}
se = senteval.engine.SE(params, batcher, prepare)
results = se.eval(transfer_tasks)
breakpoint()
print(results)

