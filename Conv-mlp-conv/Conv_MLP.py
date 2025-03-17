import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper modules for reshaping
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unflatten(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)

class ConvMLPConv(nn.Module):
    def __init__(self, 
                 conv_channels=[16, 32, 64],
                 mlp_features=[512, 512],
                 input_channels=3,
                 output_channels=1,  # New parameter
                 input_size=32,
                 dropout_rate=0.2):
        super().__init__()
        
        # Encoder with nn.Sequential
        layers = []
        in_channels = input_channels
        current_size = input_size
        for out_channels in conv_channels:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            in_channels = out_channels
            current_size //= 2
        self.encoder = nn.Sequential(*layers)
        
        # Flatten encoder output
        self.flatten = Flatten()
        flattened_size = conv_channels[-1] * current_size * current_size
        
        # MLP with nn.Sequential
        mlp_layers = []
        prev_features = flattened_size
        for feature in mlp_features:
            mlp_layers += [
                nn.Linear(prev_features, feature),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ]
            prev_features = feature
        # Output layer to map back to flattened size
        mlp_layers.append(nn.Linear(prev_features, flattened_size))
        mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Decoder with nn.Sequential
        # Modified to use output_channels for the final layer
        decoder_channels = [conv_channels[-1]] + list(reversed(conv_channels[:-1])) + [output_channels]
        decoder_layers = []
        for i in range(len(decoder_channels) - 1):
            decoder_layers.append(
                nn.ConvTranspose2d(
                    decoder_channels[i], decoder_channels[i+1],
                    kernel_size=4, stride=2, padding=1
                )
            )
            # All but the last layer get BN and ReLU
            if i != len(decoder_channels) - 2:
                decoder_layers.append(nn.BatchNorm2d(decoder_channels[i+1]))
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Unflatten to convert MLP output to 2D feature map for decoder
        self.unflatten = Unflatten(conv_channels[-1], current_size, current_size)

        
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.mlp(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x

