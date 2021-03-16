import numpy as np
import random
import os, errno
import sys
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizerFast, Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import pytorch_lightning as pl


from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from librespeech_token_dataset import *



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


    wandb_logger = WandbLogger()
    wandb_logger.experiment.init(project='cs224s-project', entity=WANDB_NAME, name=ckpt_dir, 
                config=config, sync_tensorboard=True)

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



class LightningWavyBert(pl.LightningModule):
    """PyTorch Lightning class for training a BERT-QA model."""
    def __init__(self, datapath = "/Users/jbrown/Documents/CS224NS_Data", batch_size = 16, p = 0.15, lr = 4e-5,
                        lr_decay = 0.96, weight_decay = 1e-5, lambda_AC = 1, lambda_LM = 0.2, max_label_length = 400,
                        max_w2v_length = 400, num_warmup_steps = 8000, num_train_steps = 42000):

        super().__init__()

        self.datapath = datapath
        self.batch_size = batch_size

        self.w2v_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        self.w2v_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        self.fully_connected_layer = nn.Sequential(nn.Linear(767, 768), nn.ReLU(), nn.Linear(768 ,768))
        self.fc_head = nn.Linear(self.w2v_model.lm_head.in_features, self.w2v_model.lm_head.out_features)
        self.bert_head = nn.Linear(self.w2v_model.lm_head.in_features, self.w2v_model.lm_head.out_features)
        

        self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        for param in self.roberta_model.parameters():
            param.requires_grad = False
        self.p = p
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.lambda_AC = lambda_AC 
        self.lambda_LM = lambda_LM 
        self.max_label_length = max_label_length 
        self.max_w2v_length = max_w2v_length 
        self.num_warmup_steps = num_warmup_steps
        self.num_train_steps = num_train_steps
        self.train_dataset, self.val_dataset, self.test_dataset = \
            self.create_datasets()

    def forward(self, audio_features, audio_features_length, text_labels, text_labels_length, p = 0.15):
        
        inputs = self.roberta_tokenizer(text_labels, return_tensors="pt")
        max_label_length = min(max(text_labels_length), self.max_label_length)
        inputs = list(map(torch.tensor, inputs))
        pad_inputs = torch.nn.utils.rnn.pad_sequence(a, batch_first  = True, padding_value=self.roberta_tokenizer.pad_token_id)
        roberta_outputs = self.roberta_model(**pad_inputs)
        bert_embeddings = roberta_outputs.last_hidden_state
        
        input_values = self.w2v_tokenizer(audio_features, return_tensors="pt", padding=True, max_length=self.max_w2v_length).input_values
        w2v_output = self.w2v_model(input_values)
        w2v_hidden_states = w2v_output.last_hidden_state
        cif_embeddings, n_hat = get_cif_embeddings(w2v_hidden_states, text_labels_length)
        fc_embedding = self.fully_connected_layer(cif_embeddings)
        
        bert_mixing = embedding_mixing(fc_embeddings, bert_embeddings, p = p)
        return bert_mixing, w2v_hidden_states, fc_embedding, n_hat

    def get_cif_embeddings(self, hidden_states, text_labels_length):
        max_label_length = min(max(text_labels_length), self.max_label_length)

        alpha = torch.index_select(hidden_states, 1, hidden_states.size(1)-1)
        alpha = F.sigmoid(alpha)
        scale = text_labels_length / alpha.sum(1) 
        alpha_prime *= scale

        hs = hidden_states[:,:-1,:]
        a_r = torch.zeros_like(alpha_prime)
        a_a = torch.zeros_like(alpha_prime)
        s_r = torch.zeros(hidden_states.size(0), hidden_states.size(0)-1, max_label_lengths)
        s_a = torch.zeros(hidden_states.size(0), hidden_states.size(0)-1, max_label_lengths)
        
        for k in range(alpha_prime.size(1)):
            a_a[:, k]  += alpha_prime[:, k] + a_r[:, k - 1] 
            not_fired = a_a[:, k]<1 
            a_r[not_fired, k] = a_a[not_fired, k]
            s_a[not_fired, :, k] = s_r[not_fired, :, k-1] + alpha_prime[not_fired, k] * hs[not_fired, :, k]
            s_r[not_fired, :, k] = s_a[not_fired, :, k]

            has_fired = 1 - not_fired 
            a_r[has_fired, k] = alpha_prime[has_fired, k] - (1 - a_r[has_fired, k-1])
            s_a[has_fired, :, k] = s_r[has_fired, :, k-1] + (1 - a_r[has_fired, k-1]) * hs[has_fired, :, k]
            s_r[has_fired, :, k] = a_r[has_fired, k] * hs[has_fired, :, k]

        return s_a, alpha.sum(1)


    def embedding_mixing(self, fc_embeddings, bert_embeddings, p = 0.15):
        num_batch, _, sequence_length = fc_embedding.size(2)
        mask = torch.ones(num_batch, 1, sequence_length)*p
        mask = torch.bernoulli(mask)

        return mask * fc_embeddings + (1-mask) * bert_embeddings


    def training_step(self, batch, batch_idx):
        audio_features, audio_features_length, text_labels, text_labels_length = batch
        bert_mixing, w2v_hidden_states, fc_embedding, n_hat = self.forward(audio_features, audio_features_length, text_labels, text_labels_length, p = 0.15)
        to_vocab_0 = self.bert_head(bert_mixing).permute(1,0,2)
        to_vocab_1 = self.fc_head(fc_embedding).permute(1,0,2)

        w2v_hidden_states = self.w2v_model.dropout(w2v_hidden_states)
        to_vocab_2 = self.w2v_model.lm_head(w2v_hidden_states).permute(1,0,2)
        ctc_loss_2 = F.CTCLoss(to_vocab_2, text_labels, text_labels_length, text_labels_length)
        ctc_loss_01 = F.CTCLoss(self.lambda_LM*to_vocab_0 + self.lambda_AC*to_vocab_1, text_labels, text_labels_length, text_labels_length)
        
        self.log('train_loss',ctc_loss_2+ctc_loss_01)
        self.log('train_wav2vec2_ctc_loss', ctc_loss_2, prog_bar=True, on_step=True)
        self.log('train_AC_LM_ctc_loss', ctc_loss_01, prog_bar=True, on_step=True)

        return ctc_loss_2 + ctc_loss_01


    def create_datasets(self):
        
        train_dataset = librispeech_dataset(self.datapath,  'train-clean-100')
        dev_dataset   = librispeech_dataset(self.datapath,    'dev-clean-100')
        test_dataset  =  librispeech_dataset(self.datapath,   'test-clean-100')
        return train_dataset, dev_dataset, test_dataset

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(),
                                  lr=self.lr, weight_decay=self.weight_decay)

        lr_scheduler = linear_exponential_lr_scheduler(optim, self.num_warmup_steps, self.num_train_steps, self.lr_decay)
        return [optim], [lr_scheduler] # <-- put scheduler in here if you want to use one

    def linear_exponential_lr_scheduler(optimizer, num_warmup_steps, num_train_steps, decay_rate, last_epoch= -1):

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return self.lr*float(current_step) / float(max(1.0, num_warmup_steps))
            elif (current_step >= num_warmup_steps)*(current_step < num_warmup_steps + num_train_steps):
                return self.lr
            else:
                return self.lr * decay_rate * (current_step - num_warmup_steps + num_train_steps)


        return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

    def collate_fn(self, batch):
        waveforms = [b[0] for b in batch]
        waveforms_length = [b[1] for b in batch]
        utterances = [b[2] for b in batch]
        utterances_length = [b[3] for b in batch]
        return waveforms, waveforms_length, utterances, utterances_length

    # Overwrite VALIDATION: get next minibatch
    def validation_step(self, batch, batch_idx):
        audio_features, audio_features_length, text_labels, text_labels_length = batch
        bert_mixing, w2v_hidden_states, fc_embedding, n_hat = self.forward(audio_features, audio_features_length, text_labels, text_labels_length, p = 0)
        to_vocab_0 = self.bert_head(bert_mixing).permute(1,0,2)
        to_vocab_1 = self.fc_head(fc_embedding).permute(1,0,2)

        w2v_hidden_states = self.w2v_model.dropout(w2v_hidden_states)
        to_vocab_2 = self.w2v_model.lm_head(w2v_hidden_states).permute(1,0,2)
        ctc_loss_2 = F.CTCLoss(to_vocab_2, text_labels, text_labels_length, text_labels_length)
        ctc_loss_01 = F.CTCLoss(self.lambda_LM*to_vocab_0 + self.lambda_AC*to_vocab_1, text_labels, text_labels_length, text_labels_length)
        
        self.log('val_loss',ctc_loss_2+ctc_loss_01)
        self.log('train_AC_LM_ctc_loss', ctc_loss_01, prog_bar=True, on_step=True)

        return ctc_loss_2 + ctc_loss_01
 

    def test_step(self, batch, batch_idx):
        audio_features, audio_features_length, text_labels, text_labels_length = batch
        bert_mixing, w2v_hidden_states, fc_embedding, n_hat = self.forward(audio_features, audio_features_length, text_labels, text_labels_length, p = 0)
        to_vocab_0 = self.bert_head(bert_mixing).permute(1,0,2)
        to_vocab_1 = self.fc_head(fc_embedding).permute(1,0,2)

        w2v_hidden_states = self.w2v_model.dropout(w2v_hidden_states)
        to_vocab_2 = self.w2v_model.lm_head(w2v_hidden_states).permute(1,0,2)
        ctc_loss_2 = F.CTCLoss(to_vocab_2, text_labels, text_labels_length, text_labels_length)
        ctc_loss_01 = F.CTCLoss(self.lambda_LM*to_vocab_0 + self.lambda_AC*to_vocab_1, text_labels, text_labels_length, text_labels_length)
        
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

    