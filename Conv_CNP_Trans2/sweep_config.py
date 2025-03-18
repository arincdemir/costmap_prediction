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
            'values': [128, 256]
        },
        'dropout_rate': {
            'min': 0.11,
            'max': 0.16
        },
        'cnn_channels': {
            'values': [
                [16, 16, 16],
            ]
        },
        'encoder_hidden_dims': {
            'values': [
                [256, 256],
                [256, 256, 256],
            ]
        },
        'decoder_hidden_dims': {
            'values': [
                [128, 128, 256],
                [256,256,256]
            ]
        },
    }
}

# Parameters that shouldn't change during sweeps (defaults)
default_params = {
    't_dim': 1,
    'grid_size': 32,
    'num_epochs': 5000,
    'dataset_size': 10000,
    'early_stopping_patience': 25,
    'early_stopping_min_delta': 0.000001,
}