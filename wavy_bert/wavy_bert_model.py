# Base code pulled from
# Author: Laura Kulowski
# https://github.com/lkulowski/list_encoder_decoder/blob/master/code/lstm_encoder_decoder.py

import numpy as np
import random
import os, errno
import sys
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import RobertaTokenizerFast


class LightningWavyBert(pl.LightningModule):
    """PyTorch Lightning class for training a BERT-QA model."""
    def __init__(self,learning_rate=1e-5, batch_size=16, weight_decay=1e-5):

        super().__init__()
        self.save_hyperparameters()
        self.roberta_embedding_size = 768
        self.wav2vec_embedding_size = 768
        self.lr = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay   
        self.train_dataset, self.val_dataset= \
          self.create_datasets()

        self.model = self.create_model()
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.output=[]

    def create_model(self):
        model = wavy_bert()
        
            
        return model

    def create_datasets(self):
        
        train_contexts, train_questions, train_answers = read_squad('data/train-v2.0.json')
        val_contexts, val_questions, val_answers = read_squad('data/dev-v2.0.json')

        add_end_idx(train_answers, train_contexts)
        add_end_idx(val_answers, val_contexts)

        
        train_encodings = self.tokenizer(train_contexts, train_questions, truncation=True, padding=True)
        val_encodings = self.tokenizer(val_contexts, val_questions, truncation=True, padding=True)


        add_token_positions(train_encodings, train_answers)
        add_token_positions(val_encodings, val_answers)

        train_dataset = SquadDataset(train_encodings)
        val_dataset = SquadDataset(val_encodings)

        return train_dataset, val_dataset

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(),
                                  lr=self.lr, weight_decay=self.weight_decay)
        return [optim], [] # <-- put scheduler in here if you want to use one


    def forward(self, input_ids, attention_masks, start_positions, end_positions):
        self.output = self.model(input_ids, attention_masks=attention_masks, start_positions=start_positions, end_positions=end_positions)
        return self.output.loss, self.output.start_logits, self.output.end_logits, self.output.hidden_states[0]

    def get_primary_task_loss(self, batch):
        """Returns ASR model losses, metrics, and embeddings for a batch."""
        input_ids, attention_masks = batch[0], batch[1]
        start_positions, end_positions = batch[2], batch[3]

        
        loss, start_logits,end_logits,_  = self.forward(
              input_ids, attention_masks, start_positions, end_positions)

        return loss, start_logits,end_logits,embedding

      # Overwrite TRAIN
    def training_step(self, batch):
        loss,_,_,_ = self.get_primary_task_loss(batch)
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        return loss

      # Overwrite VALIDATION: get next minibatch
    def validation_step(self, batch):
        loss,_,_,_ = self.get_primary_task_loss(batch)
        return metrics

    
    def train_dataloader(self):
        # - important to shuffle to not overfit!
        # - drop the last batch to preserve consistent batch sizes
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                            shuffle=True, pin_memory=True, drop_last=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                            shuffle=False, pin_memory=True)
        return loader


class bilstm_encoder(nn.Module):


    def __init__(self, input_size, hidden_size, num_layers = 1):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers 
        '''
        
        super(bilstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.bilstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            bidirectional = True, bias = True, batch_first = True)

    def forward(self, batched_feature_input):
        
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        
        lstm_out, self.hidden = self.bilstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
        
        return lstm_out, self.hidden     
    
    def init_hidden(self, batch_size):
        
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, num_layers = 1):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, bias = true)
        self.linear = nn.Linear(hidden_size, input_size)           

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))     
        
        return output, self.hidden

class wavy_bert(nn.Module):
    
    def __init__(self, input_size, hidden_size):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(wavy_bert, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = bilstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)


    def train_model(self, input_tensor, target_tensor, n_epochs, target_len, batch_size, training_prediction = 'recursive', teacher_forcing_ratio = 0.5, learning_rate = 0.01, dynamic_tf = False):
        
        '''
        train lstm encoder-decoder
        
        : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor    
        : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs 
        : param target_len:                number of values to predict 
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''
        
        # initialize array of losses 
        losses = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        criterion = nn.MSELoss()

        # calculate number of batch iterations
        n_batches = int(input_tensor.shape[1] / batch_size)

        with trange(n_epochs) as tr:
            for it in tr:
                
                batch_loss = 0.
                batch_loss_tf = 0.
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0

                for b in range(n_batches):
                    # select data 
                    input_batch = input_tensor[:, b: b + batch_size, :]
                    target_batch = target_tensor[:, b: b + batch_size, :]

                    # outputs tensor
                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])

                    # initialize hidden state
                    encoder_hidden = self.encoder.init_hidden(batch_size)

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing
                    decoder_input = input_batch[-1, :, :]   # shape: (batch_size, input_size)
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len): 
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target_batch[t, :, :]

                        # predict recursively 
                        else:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            
                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target_batch[t, :, :]
                            
                            # predict recursively 
                            else:
                                decoder_input = decoder_output

                    # compute the loss 
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()
                    
                    # backpropagation
                    loss.backward()
                    optimizer.step()

                # loss for epoch 
                batch_loss /= n_batches 
                losses[it] = batch_loss

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02 

                # progress bar 
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))
                    
        return losses

    def predict(self, input_tensor, target_len):
        
        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''

        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(1)     # add in batch size of 1
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden
        
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output
            
        np_outputs = outputs.detach().numpy()
        
        return np_outputs