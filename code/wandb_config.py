import wandb

sweep_config = {
    'project': 'lora-emotion-classification',
    'method': 'random',
    'metric': {
      'name': 'eval/accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-3]
        },
        'weight_decay': {
            'values': [0.01, 0.001]
        },
        'r': {
            'values': [2, 8, 32]
        },
        'lora_alpha': {
            'values': [16]
        },
        'lora_dropout': {
            'values': [0.1]
        },
        'lora_bias': {
            'values': ["all", "none", "lora_only"]
        },
        'lora_use_linear_layers': {
            'values': [True, False]
        },
        'batch_size': {
            'value': 16
        }
    }
}

