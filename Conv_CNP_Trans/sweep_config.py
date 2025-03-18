sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'validation_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.001,
            'max': 0.0011
        },
        'batch_size': {
            'values': [64]
        },
        'latent_dim': {
            'values': [394, 512]
        },
        'dropout_rate': {
            'min': 0.11,
            'max': 0.15
        },
        'cnn_channels': {
            'values': [
                [16, 16, 16],
                [24, 24, 24]
            ]
        },
        'encoder_hidden_dims': {
            'values': [
                [394, 394],
                [394, 394, 394],
            ]
        },
        'decoder_hidden_dims': {
            'values': [
                [394, 394, 394],
                [512, 512, 512],
            ]
        },
    }
}

# Parameters that shouldn't change during sweeps (defaults)
default_params = {
    't_dim': 1,
    'grid_size': 32,
    'num_epochs': 3000,
    'dataset_size': 10000,
    'early_stopping_patience': 25,
    'early_stopping_min_delta': 0.000001,
}