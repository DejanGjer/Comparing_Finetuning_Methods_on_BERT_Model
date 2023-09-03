import wandb

sweep_config = {
    'project': 'lora-emotion-classification',
    'method': 'grid',
    'metric': {
      'name': 'eval/accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [3e-4, 1e-3]
        },
        'weight_decay': {
            'values': [0.001]
        },
        'lora_use_linear_layers': {
            'values': [True, False]
        }
    }
}

