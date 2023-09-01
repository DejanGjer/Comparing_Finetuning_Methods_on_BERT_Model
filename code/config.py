
# lora
use_lora = True
r = 8
lora_alpha = 16
lora_dropout = 0.1
lora_bias = "all"
lora_use_linear_layers = False
# sampling
use_sampling = False
sample_size = 160

# model
model_name = 'bert-base-uncased'
# training
max_epochs = 3 
early_stopping_patience = 4
log_steps = 250

# test
do_test = True
num_samples_to_show = 10

# save directory
root_save_dir = './results'
save_checkpoint_limit = 2

# seed
data_seed = 42
