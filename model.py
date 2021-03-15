class speech

class LightningTextBERTQA(pl.LightningModule):
    """PyTorch Lightning class for training a BERT-QA model."""
    def __init__(self,learning_rate=1e-5, batch_size=16, weight_decay=1e-5):

        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay   
        self.train_dataset, self.val_dataset= \
          self.create_datasets()

        self.model = self.create_model()
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.output=[]

    def create_model(self):
        model = RobertaForQuestionAnswering.from_pretrained('roberta-base')
        for param in model.roberta.parameters():
            param.requires_grad = False   
            
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



class LightningTextBERTQA(pl.LightningModule):
    """PyTorch Lightning class for training a BERT-QA model."""
    def __init__(self,learning_rate=1e-5, batch_size=16, weight_decay=1e-5):

        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay   
        self.train_dataset, self.val_dataset= \
          self.create_datasets()

        self.model = self.create_model()
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.output=[]

    def create_model(self):
        model = RobertaForQuestionAnswering.from_pretrained('roberta-base')
        for param in model.roberta.parameters():
            param.requires_grad = False   
            
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


    
class LightningTextBERTQA(pl.LightningModule):
    """PyTorch Lightning class for training a BERT-QA model."""
    def __init__(self,learning_rate=1e-5, batch_size=16, weight_decay=1e-5):

        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay   
        self.train_dataset, self.val_dataset= \
          self.create_datasets()

        self.model = self.create_model()
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.output=[]

    def create_model(self):
        model = RobertaForQuestionAnswering.from_pretrained('roberta-base')
        for param in model.roberta.parameters():
            param.requires_grad = False   
            
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



class LightningTextBERTQA(pl.LightningModule):
    """PyTorch Lightning class for training a BERT-QA model."""
    def __init__(self,learning_rate=1e-5, batch_size=16, weight_decay=1e-5):

        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay   
        self.train_dataset, self.val_dataset= \
          self.create_datasets()

        self.model = self.create_model()
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.output=[]

    def create_model(self):
        model = RobertaForQuestionAnswering.from_pretrained('roberta-base')
        for param in model.roberta.parameters():
            param.requires_grad = False   
            
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

