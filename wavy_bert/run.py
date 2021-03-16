from wavy_bert import *

WANDB_NAME = 'jefe-jeff' 

config = {
    'datapath':"/Users/jbrown/Documents/CS224NS_Data",
    'p' :0.15,
    'lr' : 4e-5, 
    'lr_decay' : 0.96,
    'weight_decay' : 1e-5,
    'lambda_AC': 1,
    'lambda_LM' : 0.2,
    'max_label_length' : 400,
    'max_w2v_length' : 400,
    'num_warmup_steps' : 8000,
    'num_train_steps' :42000,
    'batch_size': 16
}

run(system="LightningWavyBert", config=config, ckpt_dir='wavy-bert', epochs=20, 
    use_gpu=True)