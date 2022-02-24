import os
import numpy as np
from dotmap import DotMap
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision

from src.datasets import datasets
from src.models.resnet import resnet18
from src.objectives.memory import MemoryBank
from src.models.logreg import LogisticRegression
from src.objectives.instdisc import InstDisc, NCE, SimCLR
from src.utils import utils
from src.utils.utils import l2_normalize


import transformers as t
from transformers import RobertaConfig, RobertaModel, BertConfig, BertModel, LongformerConfig, LongformerModel

import pytorch_lightning as pl
torch.autograd.set_detect_anomaly(True)


def create_dataloader(dataset, config, shuffle=True):
    loader = DataLoader(
        dataset, 
        batch_size=config.optim_params.batch_size,
        shuffle=shuffle, 
        pin_memory=True,
        num_workers=config.data_loader_workers,
    )
    return loader


class PretrainSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        #self.device = f'cuda:{config.gpu_device}' if config.cuda else 'cpu'
        self.train_dataset, self.val_dataset = datasets.get_datasets(
            config.data_params.dataset, is_wikipedia=True)
       
        #train_labels = self.train_dataset.dataset.targets
        #self.train_ordered_labels = np.array(train_labels)
        self.model = self.create_encoder()
#        self.memory_bank = MemoryBank(len(self.train_dataset), 
#                                      self.config.model_params.out_dim, 
#                                      device=self.device)

    def create_encoder(self):
        
        #configuration = RobertaConfig(hidden_size=self.config.model_params.out_dim, num_hidden_layers=2, num_attention_heads=2, intermediate_size=384, vocab_size=50265)
        #model = RobertaModel(configuration)
        #breakpoint()
        #configuration = LongformerConfig(hidden_size=self.config.model_params.out_dim, num_hidden_layers=2, num_attention_heads=2, intermediate_size=384, vocab_size=50265)
        config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        config.num_attention_heads = 4
        config.intermediate_size = 3072 
        config.num_hidden_layers = 4
        config.attention_window = 16
        config.hidden_size = self.config.model_params.out_dim
        model = LongformerModel(config)
        
        return model

    def configure_optimizers(self):
        #optim = torch.optim.SGD(self.model.parameters(),
        #                        lr=self.config.optim_params.learning_rate,
        #                        momentum=self.config.optim_params.momentum,
        #                        weight_decay=self.config.optim_params.weight_decay)
        optim = torch.optim.AdamW(self.model.parameters())
        return [optim], []

    def forward(self, txt):
        last_hidden_states = self.model(txt)[0]  #shape: batch_size, sequence_length, out_dim
        #average over seq_len
        average_hidden_states = last_hidden_states.mean(1) #shape: batch_size, out_dim
        return average_hidden_states

    def get_losses_for_batch(self, batch, train=True):
        indices, txt = batch
        outputs = self.forward(txt)
        loss_fn = NCE(indices, outputs, self.memory_bank,
                           k=self.config.loss_params.k,
                           t=self.config.loss_params.t,
                           m=self.config.loss_params.m)
        loss = loss_fn.get_loss()

        if train:
            with torch.no_grad():
                new_data_memory = loss_fn.updated_new_data_memory()
                self.memory_bank.update(indices, new_data_memory)

        return loss

    def get_nearest_neighbor_label(self, batch):
        """
        NOTE: ONLY TO BE USED FOR VALIDATION.

        For each image in validation, find the nearest image in the 
        training dataset using the memory bank. Assume its label as
        the predicted label.
        """
        
        #breakpoint()        
        orig, same, diff, neighbor, notneighbor = batch
        orig_output = self.forward(orig)
        same_output = self.forward(same)
        diff_output = self.forward(diff)

        neighbor_output = self.forward(neighbor)
        notneighbor_output = self.forward(notneighbor)
        
        return 0, 100
    
    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, train=True)
        metrics = {'loss': loss}
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        #breakpoint()
        #loss = self.get_losses_for_batch(batch, train=False)
        loss = self.get_val_losses_for_batch(batch, train=False)
        batch_size, num_correct_same_diff, num_correct_neighbor_not = self.get_nearest_neighbor_label(batch)
        output = OrderedDict({'val_loss': loss,
                              'val_num_correct_same_diff': num_correct_same_diff,
                              'val_num_correct_neighbor_not': num_correct_neighbor_not, 
                              'val_num_total': batch_size})
        return output
    
    def validation_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
        num_correct_same_diff = sum([out['val_num_correct_same_diff'] for out in outputs])
        num_correct_neighbor_not = sum([out['val_num_correct_neighbor_not'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_acc_same_diff = num_correct_same_diff / float(num_total)
        val_acc_neighbor_not = num_correct_neighbor_not / float(num_total)
        metrics['val_acc_same_diff'] = val_acc_same_diff
        metrics['val_acc_neighbor_not'] = val_acc_neighbor_not
        #return {'val_loss': metrics['val_loss'], 'log' : metrics}

        #filename = self.global_step 
        #filepath = os.path.join(self.default_root_dir, prefix + filename + '.ckpt')
        #self._save_model(filepath)

        return {'val_loss': metrics['val_loss'], 'log': metrics, 'val_acc_same_diff': metrics['val_acc_same_diff'], 'val_acc_neighbor_not': metrics['val_acc_neighbor_not']}

    #@pl.data_loader
    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    #@pl.data_loader
    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)

class PretrainTwoViewsSystem(PretrainSystem):

    def get_losses_for_batch(self, batch, train=True):
        #breakpoint()
        indices, txt1, txt2 = batch
        outputs1 = self.forward(txt1)
        outputs2 = self.forward(txt2)
        loss_fn = SimCLR(outputs1, outputs2, 
                         t=self.config.loss_params.t)
        loss = loss_fn.get_loss()

        #if train:
        #    with torch.no_grad():
        #        new_data_memory = (outputs1 + outputs2) / 2.
        #        self.memory_bank.update(indices, new_data_memory)

        return loss

    def get_val_losses_for_batch(self, batch, train=True):
        #breakpoint()
        index, orig, same, diff, neighbor, notneighbor = batch
        outputs1 = self.forward(orig)
        outputs2 = self.forward(same)
        loss_fn = SimCLR(outputs1, outputs2, 
                         t=self.config.loss_params.t)
        loss = loss_fn.get_loss()

        #if train:
        #    with torch.no_grad():
        #        new_data_memory = (outputs1 + outputs2) / 2.
        #        self.memory_bank.update(indices, new_data_memory)

        return loss

    def get_nearest_neighbor_label(self, batch):
        """
        NOTE: ONLY TO BE USED FOR VALIDATION.
        For each image in validation, find the nearest image in the 
        training dataset using the memory bank. Assume its label as
        the predicted label.
        """
        #breakpoint()
        index, orig, same, diff, neighbor, notneighbor = batch
        orig_output = self.forward(orig)
        same_output = self.forward(same)
        diff_output = self.forward(diff)
        neighbor_output = self.forward(neighbor)
        notneighbor_output = self.forward(notneighbor)
        

        #same vs diff document score

#        orig_output = l2_normalize(orig_output, dim=1)
#        same_output = l2_normalize(same_output, dim=1)
#        diff_output = l2_normalize(diff_output, dim=1)
#        neighbor_output = l2_normalize(neighbor_output, dim=1)
#        notneighbor_output = l2_normalize(notneighbor_output, dim=1)

        same_score = torch.sum(orig_output * same_output, dim=1)
        diff_score = torch.sum(orig_output * diff_output, dim=1)
        neighbor_score = torch.sum(orig_output * neighbor_output, dim=1)
        notneighbor_score = torch.sum(orig_output * notneighbor_output, dim=1)

        same_diff_tensor = same_score - diff_score
        num_correct_same_diff = torch.sum(same_diff_tensor >= 0)
        neighbor_not_tensor = neighbor_score - notneighbor_score
        num_correct_neighbor_not = torch.sum(neighbor_not_tensor >= 0)
    
        batch_size = orig_output.size(0)   

        return batch_size, num_correct_same_diff, num_correct_neighbor_not 



class TransferSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        #self.device = f'cuda:{config.gpu_device}' if config.cuda else 'cpu'
        #breakpoint()
        self.train_dataset, self.val_dataset = datasets.get_datasets(
            config.data_params.dataset, is_wikipedia=False)
        self.encoder = self.load_pretrained_model()
        utils.frozen_params(self.encoder)
        self.model = self.create_model()

    def load_pretrained_model(self):
        
        base_dir = self.config.pretrain_model.exp_dir
        checkpoint_name = self.config.pretrain_model.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
        config_json = utils.load_json(config_path)
        config = DotMap(config_json)
        # overwrite GPU to load on same as current agent
        config.gpu_device = self.config.gpu_device 
        config.data_params.dataset = "wikipedia_2views"
        config.system = "PretrainTwoViewsSystem"
        config.model_params.out_dim = 128
        SystemClass = globals()[config.system]
        system = SystemClass(config)
        checkpoint_file = os.path.join(base_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        old_dict = checkpoint['state_dict']
        
        #remapping

        renamed_dict = {}
        for key in old_dict:
            changed_name = key.replace('.longformer', '')
            renamed_dict[changed_name] = old_dict[key]

        system.load_state_dict(renamed_dict, strict=False)

        encoder = system.model.eval()
        
        #bertmodel = BertModel.from_pretrained('bert-base-uncased')
        #bertmodel.to('cuda:'+str(self.config.gpu_device))

        #return bertmodel
        return encoder
            
    def create_model(self):
        dataset_name = self.config.data_params.dataset
        NUM_CLASS_DICT = {'cifar10': 10, 'imagenet': 1000, 'newsgroup': 20}
        model = LogisticRegression(self.encoder.config.hidden_size, NUM_CLASS_DICT[dataset_name])
        return model

    def forward(self, txt):
        batch_size = txt.size(0)
        last_hidden_states = self.encoder(txt)[0]
        average_hidden_states = last_hidden_states.mean(1) #shape: batch_size, out_dim
        embs = average_hidden_states.view(batch_size, -1)
        return self.model(embs)

    def get_losses_for_batch(self, batch):
        _, txt, label = batch
        logits = self.forward(txt)

        return F.cross_entropy(logits, label)

    def get_accuracies_for_batch(self, batch):
        _, txt, label = batch
        logits = self.forward(txt)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        preds = preds.long().cpu()
        num_correct = torch.sum(preds == label.long().cpu()).item()
        num_total = txt.size(0)
        return num_correct, num_total

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        with torch.no_grad():
            num_correct, num_total = self.get_accuracies_for_batch(batch)
            metrics = {
                'train_loss': loss,
                'train_num_correct': num_correct,
                'train_num_total': num_total,
                'train_acc': num_correct / float(num_total),
            }
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        num_correct, num_total = self.get_accuracies_for_batch(batch)
        return OrderedDict({
            'val_loss': loss,
            'val_num_correct': num_correct,
            'val_num_total': num_total,
            'val_acc': num_correct / float(num_total)
        })

    def validation_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
        num_correct = sum([out['val_num_correct'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc
        return {'val_loss': metrics['val_loss'], 'log': metrics, 'val_acc': val_acc}

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum,
            weight_decay=self.config.optim_params.weight_decay,
        )
        return [optim], []

    #@pl.data_loader
    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    #@pl.data_loader
    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)
