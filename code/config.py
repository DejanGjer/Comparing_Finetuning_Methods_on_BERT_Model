
# lora
use_lora = True
r = 2
lora_alpha = 16
lora_dropout = 0.1
lora_bias = "none"
# lora_use_linear_layers = False

# sampling
use_sampling = False
sample_size = 160

# model
model_name = 'bert-base-uncased'
# training
batch_size = 16
max_epochs = 10 
early_stopping_patience = 4
log_steps = 250

# test
do_test = True
num_samples_to_show = 50

# save directory
root_save_dir = './results'
save_checkpoint_limit = 2

# seed
data_seed = 42
