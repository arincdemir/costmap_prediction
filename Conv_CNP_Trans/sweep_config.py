sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'validation_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0001,
            'max': 0.01
        },
        'batch_size': {
            'values': [64, 128, 256, 512]
        },
        'latent_dim': {
            'values': [128, 256, 384]
        },
        'dropout_rate': {
            'min': 0.1,
            'max': 0.3
        },
        'cnn_channels': {
            'values': [
                [16, 32, 64],
                [32, 32, 64],
                [32, 64, 128]
            ]
        },
        'encoder_hidden_dims': {
            'values': [
                [128, 128],
                [256, 256],
                [128, 256]
            ]
        },
        'decoder_hidden_dims': {
            'values': [
                [128, 256, 256],
                [256, 256, 512],
                [256, 512, 512]
            ]
        },
    }
}

# Parameters that shouldn't change during sweeps (defaults)
default_params = {
    't_dim': 1,
    'grid_size': 32,
    'num_epochs': 8000,
    'dataset_size': 5120,
    'dataset_population_factor': 4,
    'early_stopping_patience': 25,
    'early_stopping_min_delta': 0.000001,
}