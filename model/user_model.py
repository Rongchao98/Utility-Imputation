import torch
import torch.nn as nn
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------- Utility Functions ----------------------------
def get_encoder(config):
    encoders = {
        'cnn': CNNEncoder,
        'lstm': LSTMEncoder,
        'transformer': TransformerEncoder,
    }
    if config['encoder_type'] not in encoders:
        raise ValueError(f"Unsupported encoder type: {config['encoder_type']}")
    return encoders[config['encoder_type']](config)

def get_decoder(config):
    decoders = {
        'cnn': CNNDecoder,
        'transformer': TransformerDecoder,
    }
    if config['decoder_type'] not in decoders:
        raise ValueError(f"Unsupported decoder type: {config['decoder_type']}")
    return decoders[config['decoder_type']](config)

# ---------------------------- VAEs ----------------------------
class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config

        # Instantiate encoder and decoder
        self.encoder = get_encoder(config)
        self.decoder = get_decoder(config)

        # Store latent space dimensionality
        self.latent_dim = config['latent_dim']

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) using N(0, 1).

        :param mu: Mean tensor.
        :param log_var: Log variance tensor.
        :return: Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # Sample from standard normal
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE.

        :param x: Input tensor (e.g., time series data).
        :return: Reconstructed output, mean, log variance, and latent vector.
        """
        # Encode input to latent space
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        # Decode latent vector to reconstructed output
        reconstructed = self.decoder(z)

        # loss
        total_loss, reconstruction_loss, kl_div = self.loss_function(reconstructed, x, mu, log_var)

        return reconstructed, total_loss

    def loss_function(self, reconstructed, x, mu, log_var):
        """
        Compute the VAE loss (reconstruction loss + KL divergence).

        :param reconstructed: Reconstructed output.
        :param x: Original input.
        :param mu: Mean tensor from the encoder.
        :param log_var: Log variance tensor from the encoder.
        :return: Total loss, reconstruction loss, and KL divergence.
        """
        # Reconstruction loss (can use MSE or BCE depending on the data)
        reconstruction_loss = F.mse_loss(reconstructed, x, reduction='mean')

        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_div /= x.size(0)  # Normalize by batch size

        # Total loss
        total_loss = reconstruction_loss + kl_div
        return total_loss, reconstruction_loss, kl_div

# ---------------------------- Encoder ----------------------------
class Encoder(torch.nn.Module, ABC):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        pass

class CNNEncoder(Encoder, nn.Module):
    def __init__(self, config):
        """
        CNN-based encoder for time series.
        """
        Encoder.__init__(self, config)
        nn.Module.__init__(self)

        input_dim = config['input_dim']  # (T, F)
        latent_dim = config['latent_dim'] # (D)
        T, F = input_dim

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=F, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.flatten_dim = T // 4 * 64  # Assuming 2 pooling layers
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        # x shape: (B, T, F)
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.conv_layers(x) # (B, 64, T // 4)
        x = x.view(x.size(0), -1)  # (B, 64 * T // 4)
        mu = self.fc_mu(x) # (B, D)
        log_var = self.fc_log_var(x) # (B, D)
        return mu, log_var

class LSTMEncoder(Encoder, nn.Module):
    def __init__(self, config):
        """
        RNN-based encoder for time series.
        """
        Encoder.__init__(self, config)
        nn.Module.__init__(self)

        input_dim = config['input_dim']  # (T, F)
        latent_dim = config['latent_dim'] # (D)
        T, F = input_dim

        self.rnn = nn.LSTM(input_size=F, hidden_size=64, num_layers=2, batch_first=True)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

    def forward(self, x):
        # x shape: (B, T, F)
        _, (h_n, _) = self.rnn(x)  # h_n shape: (num_layers, batch_size, hidden_size)
        h_n = h_n[-1]
        mu = self.fc_mu(h_n) # (B, D)
        log_var = self.fc_log_var(h_n) # (B, D)
        return mu, log_var

class TransformerEncoder(Encoder, nn.Module):
    def __init__(self, config):
        """
        Transformer-based encoder for time series.
        """
        Encoder.__init__(self, config)
        nn.Module.__init__(self)

        input_dim = config['input_dim']  # (T, F)
        latent_dim = config['latent_dim']
        num_heads = config.get('num_heads', 4)       # Default to 4 attention heads
        ff_dim = config.get('ff_dim', 128)           # Feedforward layer dimension
        num_layers = config.get('num_layers', 2)     # Number of Transformer layers
        dropout_rate = config.get('dropout', 0.1)    # Dropout rate

        T, F = input_dim
        self.embedding = nn.Linear(F, 64)  # Project input features to embedding dimension
        self.positional_encoding = self._create_positional_encoding(T, 64)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Latent space
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

    def _create_positional_encoding(self, max_len, d_model):
        """
        Create positional encoding for time series inputs.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        """
        Forward pass for the Transformer encoder.

        :param x: Input tensor of shape (B, T, F).
        :return: Mu and log_var tensors of shape (B, latent_dim).
        """
        batch_size, T, F = x.shape
        x = self.embedding(x)  # Shape: (batch_size, T, embedding_dim)
        x = x + self.positional_encoding[:, :T, :].to(x.device)  # Add positional encoding
        x = x.permute(1, 0, 2)  # Shape: (T, batch_size, embedding_dim) for Transformer

        x = self.transformer(x)  # Shape: (T, batch_size, embedding_dim)
        x = x.mean(dim=0)  # Global average pooling over time dimension (T)

        mu = self.fc_mu(x)  # Shape: (batch_size, latent_dim)
        log_var = self.fc_log_var(x)  # Shape: (batch_size, latent_dim)
        return mu, log_var

# ---------------------------- Decoder ----------------------------
class Decoder(torch.nn.Module, ABC):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        pass

class CNNDecoder(Decoder, nn.Module):
    def __init__(self, config):
        """
        CNN-based decoder for time series reconstruction.
        """
        super().__init__(config)
        nn.Module.__init__(self)

        latent_dim = config['latent_dim']
        output_dim = config['input_dim'][1]  # F
        T = config['input_dim'][0]  # T

        # Fully connected layer to expand latent vector
        self.fc = nn.Linear(latent_dim, 64 * (T // 4))  # Adjust based on your network

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, output_dim, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, z):
        """
        Forward pass of the CNN decoder.
        :param z: Latent space representation.
        :return: Reconstructed input.
        """
        x = self.fc(z)  # (batch_size, latent_dim) -> (batch_size, 64 * (T // 4))
        x = x.view(x.size(0), 64, -1)  # Reshape to (batch_size, 64, T // 4)
        x = self.deconv_layers(x)  # Upsample to original size (batch_size, F, T)
        x = x.permute(0, 2, 1)  # Convert to (batch_size, T, F)
        return x

class TransformerDecoder(Decoder, nn.Module):
    def __init__(self, config):
        """
        Transformer-based decoder for time series reconstruction.
        """
        super().__init__(config)
        nn.Module.__init__(self)

        latent_dim = config['latent_dim']
        output_dim = config['input_dim'][1]  # F
        T = config['input_dim'][0]  # T
        num_heads = config.get('num_heads', 4)
        ff_dim = config.get('ff_dim', 128)
        num_layers = config.get('num_layers', 2)
        dropout_rate = config.get('dropout', 0.1)

        # Fully connected layer to project the latent vector
        self.fc = nn.Linear(latent_dim, 64)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(T, 64)

        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=64, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout_rate
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final linear layer to map output to original feature dimension
        self.fc_out = nn.Linear(64, output_dim)

    def _create_positional_encoding(self, max_len, d_model):
        """
        Create positional encoding for time series inputs.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, z):
        """
        Forward pass of the Transformer decoder.
        :param z: Latent space representation.
        :return: Reconstructed input.
        """
        batch_size = z.size(0)
        T = self.config['input_dim'][0]  # Number of time steps

        # Project latent vector into a sequence of vectors for transformer
        x = self.fc(z).unsqueeze(0)  # Shape: (1, batch_size, 64)
        x = x.repeat(T, 1, 1)  # Repeat for T time steps, Shape: (T, batch_size, 64)

        # Add positional encoding
        x = x + self.positional_encoding[:T, :].to(x.device)

        # Decoder with Transformer
        x = self.transformer_decoder(x, x)  # Self-attention, no encoder (only decoder)
        x = x.mean(dim=0)  # Global average pooling (or use the final step)

        # Output layer to reconstruct the input
        x = self.fc_out(x)  # Shape: (batch_size, T, F)
        return x