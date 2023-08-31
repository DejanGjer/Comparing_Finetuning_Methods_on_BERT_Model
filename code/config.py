
# lora
use_lora = False
r = 8
lora_alpha = 16
lora_dropout = 0.1
# sampling
use_sampling = True
sample_size = 160
# training
max_epochs = 2 
early_stopping_patience = 4
log_steps = 2
# model
model_name = 'bert-base-uncased'
# save directory
root_save_dir = './results'
save_checkpoint_limit = 2

# seed
data_seed = 42
