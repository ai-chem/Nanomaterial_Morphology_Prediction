from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import nn
import torch
from torch.nn import functional as F

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
    
class VAE(LightningModule):

    def __init__(
        self,
        input_height: int,
        enc_type: str = "default",
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 4096,
        kl_coeff: float = 0.01,
        latent_dim: int = 4096,
        lr: float = 1e-3,
        **kwargs,
    ):
        """
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        
        self.input_height = input_height
        self.latent_dim = latent_dim
    
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 128 x 64 x 64

            nn.Conv2d(128, 64, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 64 x 32 x 32

            nn.Conv2d(64, 32, (3,3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
            nn.ReLU(True),

            nn.Conv2d(32, 4, (3, 3), stride=(1,1), padding=(1,1)),  # 4 x 32 x 32
            nn.Flatten()  # 4096 x 1 x 1
        )

        self.decoder = nn.Sequential(    
            View([-1, 4, 32, 32]),
            nn.ConvTranspose2d(4, 32, (3, 3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 32 x 64 x 64

            nn.ConvTranspose2d(32, 64, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 64 x 128 x 128

            nn.ConvTranspose2d(64, 128, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 1, (3,3), stride=(1,1), padding=(1,1)),
            nn.Sigmoid()
        )
        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q
    
    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)
        ls = nn.BCELoss()
        recon_loss = ls(x_hat, x)
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = recon_loss + kl 

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)