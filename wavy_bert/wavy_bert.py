import numpy as np
import random
import os, errno
import sys
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizerFast, Wav2Vec2Tokenizer, Wav2Vec2Model, Wav2Vec2Config
import pytorch_lightning as pl

import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from librespeech_token_dataset import *

from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def run(system, config, ckpt_dir, epochs=1, monitor_key='val_loss', 
        use_gpu=False, seed=1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    SystemClass = globals()[system]
    system = SystemClass(**config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("/Users/jbrown/Documents/CS224NS_Models", ckpt_dir),
        save_top_k=1,
        verbose=True,
        monitor=monitor_key, 
        mode='min')

    wandb.init(project='cs224s-project', entity='jefe-jeff', name=ckpt_dir, 
             config=config, sync_tensorboard=True)
    
    wandb_logger = WandbLogger()


    if use_gpu:
        trainer = pl.Trainer(
            gpus=1, max_epochs=epochs, min_epochs=epochs,
            checkpoint_callback=checkpoint_callback, logger=wandb_logger)
    else:
        trainer = pl.Trainer(
            max_epochs=epochs, min_epochs=epochs,
            checkpoint_callback=checkpoint_callback, logger=wandb_logger)

    trainer.fit(system)
    result = trainer.test()

class WavyBert(nn.Module):

    def __init__(self, lambda_AC = 1, lambda_LM = 0.2, max_label_length = 400,
                        max_w2v_length = 400):
        super().__init__()

        self.lambda_LM = lambda_LM
        self.lambda_AC = lambda_AC
        self.max_label_length = max_label_length
        self.max_w2v_length = max_w2v_length

        self.w2v_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        self.w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.configuration = Wav2Vec2Config()
        self.vocab_size = self.configuration.vocab_size
        self.hidden_size = self.configuration.hidden_size
        self.dropout = nn.Dropout(p=0.1)

        self.fully_connected_layer = nn.Sequential(nn.Linear(self.hidden_size-1, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size ,self.hidden_size))
        self.fc_head = nn.Linear(self.hidden_size, self.vocab_size)
        self.bert_head = nn.Linear(self.hidden_size, self.vocab_size)
        self.w2v_head = nn.Linear(self.hidden_size, self.vocab_size)
        

        self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        for param in self.roberta_model.parameters():
            param.requires_grad = False


    def forward(self, audio_features, audio_features_length, text_labels, text_labels_length, p = 0.15, split = 'train'):
        # `torch.nn.utils.rnn.pack_padded_sequence` collapses padded sequences
        # to a contiguous chunk
        parameter = next(self.roberta_model.parameters())

        
        bert_input_ids = self.roberta_tokenizer(text_labels).input_ids
        
        max_label_length = min(max(text_labels_length), self.max_label_length)
        
        text_labels_length = torch.tensor(text_labels_length).type_as(parameter).int()
        
        
        
        audio_features_length = torch.tensor(audio_features_length).type_as(parameter).int()

        bert_input_ids = list(map(torch.tensor, bert_input_ids))
        
        bert_input_ids_length = torch.tensor([len(b) for b in bert_input_ids]).type_as(parameter).int()

        
        pad_bert_input_ids = torch.nn.utils.rnn.pad_sequence(bert_input_ids, batch_first  = True,
                  padding_value=self.roberta_tokenizer.pad_token_id).type_as(parameter)
        pad_bert_input_ids = pad_bert_input_ids.int()
        roberta_outputs = self.roberta_model(pad_bert_input_ids)
        bert_embeddings = roberta_outputs.last_hidden_state
  

        audio_features = audio_features
        input_values = self.w2v_tokenizer(audio_features, return_tensors="pt", padding=True).input_values.type_as(parameter)
        w2v_output = self.w2v_model(input_values, return_dict=True)
       
        w2v_hidden_states = w2v_output['last_hidden_state']
        cif_embeddings, n_hat = self.get_cif_embeddings(w2v_hidden_states, bert_input_ids_length)

        fc_embeddings = self.fully_connected_layer(cif_embeddings)
        
        bert_mixing = self.embedding_mixing(fc_embeddings, bert_embeddings, p = p)

        return bert_mixing, w2v_hidden_states, fc_embeddings, n_hat
     
    def get_cif_embeddings(self, hidden_states, text_labels_length, split = 'train'):
        
        if split == 'train':

            max_label_length = min(max(text_labels_length), self.max_label_length)
            text_labels_length = torch.clone(text_labels_length)
            text_labels_length[text_labels_length > max_label_length] = max_label_length
            alpha = hidden_states[:,:,-1]
            alpha = F.sigmoid(alpha)
            
            scale = text_labels_length / alpha.sum(1) 
            alpha_prime = alpha * scale.unsqueeze(1)

            hs = hidden_states[:,:,:-1].type_as(hidden_states)
            a_r = torch.zeros_like(alpha_prime).type_as(hidden_states)
            a_a = torch.zeros_like(alpha_prime).type_as(hidden_states)
            s_r = torch.zeros(hidden_states.size(0), max_label_length, hidden_states.size(2)-1).type_as(hidden_states)
            s_a = torch.zeros(hidden_states.size(0), max_label_length, hidden_states.size(2)-1).type_as(hidden_states)
            
            for k in range(max_label_length):
                a_a[:, k]  += alpha_prime[:, k] + a_r[:, k - 1] 
                not_fired = a_a[:, k]<1 
                a_r[not_fired, k] = a_a[not_fired, k]

                s_a[not_fired, k, :] = s_r[not_fired, k-1, :] + alpha_prime[not_fired, k].unsqueeze(1) * hs[not_fired, k, :]
                s_r[not_fired, k, :] = s_a[not_fired, k, :]

                has_fired = (1 - not_fired.int() ).bool()
                a_r[has_fired, k] = alpha_prime[has_fired, k] - (1 - a_r[has_fired, k-1])
                s_a[has_fired, k, :] = s_r[has_fired, k-1, :] + (1 - a_r[has_fired, k-1]).unsqueeze(1) * hs[has_fired, k, :]
                s_r[has_fired, k, :] = a_r[has_fired, k].unsqueeze(1) * hs[has_fired, k, :]

            return s_a, (text_labels_length - alpha.sum(1)).pow(2).sum()
            
        else:
            alpha = hidden_states[:,:,-1]
            alpha_prime = F.sigmoid(alpha)
            
            max_label_length = (max(alpha_prime.sum(2))+1)//1

            hs = hidden_states[:,:,:-1].type_as(hidden_states)
            a_r = torch.zeros_like(alpha_prime).type_as(hidden_states)
            a_a = torch.zeros_like(alpha_prime).type_as(hidden_states)
            s_r = torch.zeros(hidden_states.size(0), max_label_length, hidden_states.size(2)-1).type_as(hidden_states)
            s_a = torch.zeros(hidden_states.size(0), max_label_length, hidden_states.size(2)-1).type_as(hidden_states)
            
            for k in range(max_label_length):
                a_a[:, k]  += alpha_prime[:, k] + a_r[:, k - 1] 
                not_fired = a_a[:, k]<1 
                a_r[not_fired, k] = a_a[not_fired, k]
                s_a[not_fired, k, :] = s_r[not_fired, k-1, :] + alpha_prime[not_fired, k].unsqueeze(1) * hs[not_fired, k, :]
                s_r[not_fired, k, :] = s_a[not_fired, k, :]

                has_fired = (1 - not_fired.int() ).bool()
                a_r[has_fired, k] = alpha_prime[has_fired, k] - (1 - a_r[has_fired, k-1])
                s_a[has_fired, k, :] = s_r[has_fired, k-1, :] + (1 - a_r[has_fired, k-1]).unsqueeze(1) * hs[has_fired, k, :]
                s_r[has_fired, k, :] = a_r[has_fired, k].unsqueeze(1) * hs[has_fired, k, :]

            return s_a, ((alpha_prime.sum(1)+1)//1 - alpha_prime.sum(1)).pow(2).sum()



    def embedding_mixing(self, fc_embeddings, bert_embeddings, p = 0.15):

        num_batch, sequence_length, _ = fc_embeddings.shape
        mask = p*torch.ones(num_batch, sequence_length, 1)
        mask = torch.bernoulli(mask).type_as(fc_embeddings)
        print(fc_embeddings.shape)
        print(bert_embeddings.shape)
        return (1-mask) * fc_embeddings + (mask) * bert_embeddings

  


class LightningWavyBert(pl.LightningModule):
    """PyTorch Lightning class for training a BERT-QA model."""
    def __init__(self, datapath = "/Users/jbrown/Documents/CS224NS_Data", batch_size = 16, p = 0.15, lr = 4e-5,
                        lr_decay = 0.96, weight_decay = 1e-5, lambda_AC = 1, lambda_LM = 0.2, max_label_length = 400,
                        max_w2v_length = 400, num_warmup_steps = 8000, num_train_steps = 42000):

        super().__init__()

     

        self.datapath = datapath
        self.batch_size = batch_size

        self.lambda_AC = lambda_AC 
        self.lambda_LM = lambda_LM 
        self.max_label_length = max_label_length 
        self.max_w2v_length = max_w2v_length 

        self.p = p
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        
        self.num_warmup_steps = num_warmup_steps
        self.num_train_steps = num_train_steps
        self.train_dataset, self.val_dataset, self.test_dataset = \
            self.create_datasets()

        self.model = self.create_model()
        self.save_hyperparameters()

    def create_model(self):
        model = WavyBert(self.lambda_AC, self.lambda_LM, self.max_label_length,
                        self.max_w2v_length
          )
        return model

    def forward(self, audio_features, audio_features_length, text_labels, text_labels_length, p = 0.15, split = 'train'):
        bert_mixing, w2v_hidden_states, fc_embedding, n_hat =self.model(audio_features, 
                                                                        audio_features_length, text_labels, text_labels_length, p=p, split = split)
        return bert_mixing, w2v_hidden_states, fc_embedding, n_hat

    def training_step(self, batch, batch_idx):
        audio_features, audio_features_length, text_labels, text_labels_length = batch
        bert_mixing, w2v_hidden_states, fc_embedding, n_hat = self.forward(audio_features, audio_features_length, 
                                                                          text_labels, text_labels_length, p = 0.8, split = 'train')
        to_vocab_0 = self.model.bert_head(bert_mixing).permute(1,0,2)
        to_vocab_1 = self.model.fc_head(fc_embedding).permute(1,0,2)

        w2v_hidden_states = self.model.dropout(w2v_hidden_states)
        to_vocab_2 = self.model.w2v_head(w2v_hidden_states).permute(1,0,2)

        ctc_loss = nn.CTCLoss()
        ctc_loss_2 = ctc_loss(to_vocab_2, text_labels, text_labels_length, text_labels_length)
        ctc_loss_01 = ctc_loss(self.lambda_LM*to_vocab_0 + self.lambda_AC*to_vocab_1, text_labels, text_labels_length, text_labels_length)
        
        self.log('train_loss',ctc_loss_2+ctc_loss_01)
        self.log('train_wav2vec2_ctc_loss', ctc_loss_2, prog_bar=True, on_step=True)
        self.log('train_AC_LM_ctc_loss', ctc_loss_01, prog_bar=True, on_step=True)

        return ctc_loss_2 + ctc_loss_01


    def create_datasets(self):
        train_dataset = librispeech_dataset(self.datapath,  'train-clean-100', download = True)
        dev_dataset   = librispeech_dataset(self.datapath,    'dev-clean', download = True)
        test_dataset  = librispeech_dataset(self.datapath,   'test-clean', download = True)
        return train_dataset, dev_dataset, test_dataset

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(),
                                  lr=self.lr, weight_decay=self.weight_decay)

        lr_scheduler = self.linear_exponential_lr_scheduler(optim, self.num_warmup_steps, self.num_train_steps, self.lr_decay)
        return [optim], [lr_scheduler] # <-- put scheduler in here if you want to use one

    def linear_exponential_lr_scheduler(self, optimizer, num_warmup_steps, num_train_steps, decay_rate, last_epoch= -1):

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return self.lr*float(current_step) / float(max(1.0, num_warmup_steps))
            elif (current_step >= num_warmup_steps)*(current_step < num_warmup_steps + num_train_steps):
                return self.lr
            else:
                return self.lr * decay_rate * (current_step - num_warmup_steps + num_train_steps)


        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

    def collate_fn(self, batch):
        waveforms = [b[0] for b in batch]
        waveforms_length = [b[1] for b in batch]
        utterances = [b[2] for b in batch]
        utterances_length = [b[3] for b in batch]
        return waveforms, waveforms_length, utterances, utterances_length

    # Overwrite VALIDATION: get next minibatch
    def validation_step(self, batch, batch_idx):
        audio_features, audio_features_length, text_labels, text_labels_length = batch
        
        bert_input_ids = self.model.roberta_tokenizer(text_labels).input_ids
        bert_input_ids = list(map(torch.tensor, bert_input_ids))
        bert_input_ids_length = tuple([len(b) for b in bert_input_ids])

        pad_bert_input_ids = torch.nn.utils.rnn.pad_sequence(bert_input_ids, batch_first  = True,
                  padding_value=self.model.roberta_tokenizer.pad_token_id).int()
        

        bert_mixing, w2v_hidden_states, fc_embedding, n_hat = self.forward(audio_features, audio_features_length, text_labels, text_labels_length, p = 0, split = 'val')
        to_vocab_0 = self.model.bert_head(bert_mixing).permute(1,0,2)
        to_vocab_1 = self.model.fc_head(fc_embedding).permute(1,0,2)

        w2v_hidden_states = self.model.dropout(w2v_hidden_states)
        to_vocab_2 = self.model.w2v_head(w2v_hidden_states).permute(1,0,2)
        ctc_loss = nn.CTCLoss()
        ctc_loss_2 = ctc_loss(to_vocab_2, pad_bert_input_ids, bert_input_ids_length, bert_input_ids_length)
        ctc_loss_01 = ctc_loss(self.lambda_LM*to_vocab_0 + self.lambda_AC*to_vocab_1, pad_bert_input_ids, bert_input_ids_length, bert_input_ids_length)
        
        self.log('val_loss',ctc_loss_2+ctc_loss_01)
        self.log('train_AC_LM_ctc_loss', ctc_loss_01, prog_bar=True, on_step=True)

        return ctc_loss_2 + ctc_loss_01
 

    def test_step(self, batch, batch_idx):
        audio_features, audio_features_length, text_labels, text_labels_length = batch
        bert_mixing, w2v_hidden_states, fc_embedding, n_hat = self.forward(audio_features, audio_features_length, text_labels, text_labels_length, p = 0, split = 'val')
        to_vocab_0 = self.model.bert_head(bert_mixing).permute(1,0,2)
        to_vocab_1 = self.model.fc_head(fc_embedding).permute(1,0,2)

        w2v_hidden_states = self.model.dropout(w2v_hidden_states)
        to_vocab_2 = self.model.w2v_head(w2v_hidden_states).permute(1,0,2)
        
        ctc_loss = nn.CTCLoss()
        ctc_loss_2 = ctc_loss(to_vocab_2, text_labels, text_labels_length, text_labels_length)
        ctc_loss_01 = ctc_loss(self.lambda_LM*to_vocab_0 + self.lambda_AC*to_vocab_1, text_labels, text_labels_length, text_labels_length)
        
        self.log('val_loss',ctc_loss_2+ctc_loss_01)
        self.log('train_AC_LM_ctc_loss', ctc_loss_01, prog_bar=True, on_step=True)

        return ctc_loss_2 + ctc_loss_01


        
    def train_dataloader(self):
        # - important to shuffle to not overfit!
        # - drop the last batch to preserve consistent batch sizes
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                            shuffle=True, pin_memory=True, drop_last=True, collate_fn = self.collate_fn)
        return loader

    def val_dataloader(self):
        
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                            shuffle=False, pin_memory=True, collate_fn = self.collate_fn)

        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                            shuffle=False, pin_memory=True, collate_fn = self.collate_fn)
        
        return loader

    