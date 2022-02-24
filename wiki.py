import os; import psutil; import timeit
import random
from nlp import load_dataset
from nlp import DownloadConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
import transformers as t
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import TensorDataset
from src.datasets.wikipedia import WIKIPEDIA

conf = DownloadConfig(cache_dir='/data5/nguyenjd/wikipedia')
wiki = WIKIPEDIA(train=True) 
#wiki = load_dataset("wikipedia", "20200501.en", split='train', cache_dir='/data5/nguyenjd/wikipedia', download_config=conf)

#untokenized_loader = DataLoader(wiki, batch_size=64, shuffle=True)
loader = DataLoader(wiki, batch_size=4, sampler = sampler.RandomSampler(wiki, num_samples=8, replacement=True))

def fill_blanks(input_context):
    tokenizer = t.BertTokenizer.from_pretrained('bert-base-uncased')
    masked = input_context.replace("@", tokenizer.mask_token)
    print(masked)

    tok_ids = tokenizer.encode(masked)
    tok_ids_tensor = torch.Tensor([tok_ids])

    model = t.BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    with torch.no_grad():
        outputs = model(tok_ids_tensor.long())
        predictions = outputs[0]
    most_probable_tok_ids = torch.argmax(predictions, dim=2)
    lst = most_probable_tok_ids.flatten().tolist()
    most_probable_toks = tokenizer.decode(lst)
    print(most_probable_toks)


def tokenize_dataset(loader):
    counter = 0
    size = len(loader)
    tokenizer = t.RobertaTokenizerFast.from_pretrained('roberta-base') 
    for data in loader:
        counter+=1
        text = data['text'][0]
        tok_ids = tokenizer.encode(text, truncation=True)       
        tok_ids_tensor = torch.Tensor([tok_ids])
        articles.append(tok_ids_tensor)
        print(f"tokenized {counter:.1f} out of {size:.1f}")

def getStats(obj):
    for data in obj:
        print("size of data:", data.size)
    print(len(obj))

for data in loader:
    print(type(data))
    print(data[1])
    print((data[1]).shape)
print(wiki[6010037])
def getRandomSubstring(word):
    print(word)
    split = word.split()
    length = len(split)
    idx1 = random.randint(0, length-1)
    idx2 = random.randint(0, length-1)
    print("length = ", length)
    print("idx1 = ", idx1)
    print("idx2 = ", idx2)
    if idx1 < idx2:
        substr = split[idx1:idx2]
    else:
        substr = split[idx2:idx1]

    print(substr)
    strsent = " ".join(str(x) for x in substr)
    print(strsent)


#word = "hello, I am John. This is a test of the random span"
#getRandomSubstring(word)
