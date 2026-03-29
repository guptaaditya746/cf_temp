import pytorch_lightning as pl
import torch
import torch.nn as nn


class TimeSeriesAutoencoder(pl.LightningModule):
    def __init__(self, seq_len=24, n_features=6, hidden_dim=64, latent_dim=16, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr

        self.encoder_lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.encoder_linear = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.decoder_linear = nn.Linear(self.hidden_dim, self.n_features)
        self.criterion = nn.MSELoss()

    def encode(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        hidden = hidden.squeeze(0)
        latent = self.encoder_linear(hidden)
        return latent

    def decode(self, latent):
        latent_repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoder_out, _ = self.decoder_lstm(latent_repeated)
        reconstruction = self.decoder_linear(decoder_out)
        return reconstruction

    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction

    def training_step(self, batch, batch_idx):
        x_hat = self(batch)
        loss = self.criterion(x_hat, batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_hat = self(batch)
        loss = self.criterion(x_hat, batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


MODEL_REGISTRY = {
    "lstm_autoencoder": TimeSeriesAutoencoder,
}


def build_model(model_name, model_config):
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(
            f"Unknown model_name={model_name!r}. Available: {tuple(MODEL_REGISTRY)}"
        )
    return model_cls(**model_config)


def load_model_from_checkpoint(model_name, checkpoint_path, device):
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model
