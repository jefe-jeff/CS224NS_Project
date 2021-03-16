import torchaudio
import pytorch_lightning as pl
from wavy_bert import * 

train_dataset = torchaudio.datasets.LIBRISPEECH("/Users/jbrown/Documents/CS224NS_Data",  'train-clean-100')
dev_dataset   = torchaudio.datasets.LIBRISPEECH("/Users/jbrown/Documents/CS224NS_Data",    'dev-clean-100')
test_dataset  = torchaudio.datasets.LIBRISPEECH("/Users/jbrown/Documents/CS224NS_Data",   'test-clean-100')

def collate_fn(batch):
    waveforms = [b[0] for b in batch]
    utterances = [b[1] for b in batch]
    return waveforms, utterances

train_dataloader = DataLoader(train_dataset, collate_fn = collate_fn)
dev_dataloader = DataLoader(dev_dataset, collate_fn = collate_fn)
test_dataloader = DataLoader(test_dataset, collate_fn = collate_fn)

autoencoder = LightningWavyBert()
trainer = pl.Trainer()
trainer.fit(autoencoder, , DataLoader(dev_dataset), DataLoader(test_dataset))