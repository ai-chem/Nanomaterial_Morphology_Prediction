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
        enc_type: str = "4_layers",
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
        valid_encoders = {
            "resnet18": {
                "enc": resnet18_encoder,
                "dec": resnet18_decoder,
            },
            "resnet50": {
                "enc": resnet50_encoder,
                "dec": resnet50_decoder,
            },
        }
        if enc_type == "4_layers":
            if input_height == 128:
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
            elif input_height == 224:
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 224, (3,3), stride=(1,1), padding=(1,1)),  # 224 x 224 x 224
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 224 x 112 x 112
        
                    nn.Conv2d(224, 112, (3,3), stride=(1,1), padding=(1,1)),  # 112 x 112 x 112
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 112 x 56 x 56
        
                    nn.Conv2d(112, 56, (3,3), stride=(1,1), padding=(1,1)),  # 56 x 56 x 56
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 56 x 28 x 28
        
                    nn.Conv2d(56, 7, (3, 3), stride=(1,1), padding=(1,1)),  # 7 x 28 x 28
                    nn.Flatten()  # 5488 x 1 x 1
                )
                 
                self.decoder = nn.Sequential(    
                    View([-1, 7, 28, 28]),
                    nn.ConvTranspose2d(7, 56, (3, 3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 32 x 64 x 64
                
                    nn.ConvTranspose2d(56, 112, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 64 x 128 x 128
        
                    nn.ConvTranspose2d(112, 224, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 64 x 128 x 128
        
                    nn.ConvTranspose2d(224, 1, (3,3), stride=(1,1), padding=(1,1)),
                    nn.Sigmoid()
                )
        elif enc_type == "3_layers":
            if input_height == 128:
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 128, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 128 x 64 x 64

                    nn.Conv2d(128, 64, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 64 x 32 x 32

                    nn.Conv2d(64, 4, (3, 3), stride=(1,1), padding=(1,1)),  # 4 x 32 x 32
                    nn.Flatten()  # 4096 x 1 x 1
                )

                self.decoder = nn.Sequential(    
                    View([-1, 4, 32, 32]),
                    nn.ConvTranspose2d(4, 32, (3, 3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 32 x 64 x 64

                    nn.ConvTranspose2d(32, 128, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 64 x 128 x 128

                    nn.ConvTranspose2d(128, 1, (3,3), stride=(1,1), padding=(1,1)),
                    nn.Sigmoid()
                )
            elif input_height == 224:
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 224, (3,3), stride=(1,1), padding=(1,1)),  # 224 x 224 x 224
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 224 x 112 x 112
        
                    nn.Conv2d(224, 56, (3,3), stride=(1,1), padding=(1,1)),  # 112 x 112 x 112
                    nn.ReLU(True),
                    nn.MaxPool2d(4),  # 112 x 56 x 56
        
                    nn.Conv2d(56, 7, (3, 3), stride=(1,1), padding=(1,1)),  # 7 x 28 x 28
                    nn.Flatten()  # 5488 x 1 x 1
                )
                 
                self.decoder = nn.Sequential(    
                    View([-1, 7, 28, 28]),
                    nn.ConvTranspose2d(7, 112, (3, 3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=4),  # 32 x 64 x 64
        
                    nn.ConvTranspose2d(112, 224, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 64 x 128 x 128
        
                    nn.ConvTranspose2d(224, 1, (3,3), stride=(1,1), padding=(1,1)),
                    nn.Sigmoid()
                )
        elif enc_type == "5_layers":
            if input_height == 128:
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 128, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 128 x 64 x 64

                    nn.Conv2d(128, 64, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 64 x 32 x 32

                    nn.Conv2d(64, 32, (3,3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
                    nn.ReLU(True),

                    nn.Conv2d(32, 16, (3,3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
                    nn.ReLU(True),

                    nn.Conv2d(16, 4, (3, 3), stride=(1,1), padding=(1,1)),  # 4 x 32 x 32
                    nn.Flatten()  # 4096 x 1 x 1
                )

                self.decoder = nn.Sequential(    
                    View([-1, 4, 32, 32]),
                    nn.ConvTranspose2d(4, 16, (3, 3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 32 x 64 x 64

                    nn.ConvTranspose2d(16, 32, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 64 x 128 x 128

                    nn.ConvTranspose2d(32, 64, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
                    nn.ReLU(True),

                    nn.ConvTranspose2d(64, 128, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
                    nn.ReLU(True),

                    nn.ConvTranspose2d(128, 1, (3,3), stride=(1,1), padding=(1,1)),
                    nn.Sigmoid()
                )
            elif input_height == 224:
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 224, (3,3), stride=(1,1), padding=(1,1)),  # 224 x 224 x 224
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 224 x 112 x 112
        
                    nn.Conv2d(224, 112, (3,3), stride=(1,1), padding=(1,1)),  # 112 x 112 x 112
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 112 x 56 x 56
        
                    nn.Conv2d(112, 56, (3,3), stride=(1,1), padding=(1,1)),  # 56 x 56 x 56
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 56 x 28 x 28

                    nn.Conv2d(56, 14, (3,3), stride=(1,1), padding=(1,1)),  # 56 x 56 x 56
                    nn.ReLU(True),

                    nn.Conv2d(56, 7, (3, 3), stride=(1,1), padding=(1,1)),  # 7 x 28 x 28
                    nn.Flatten()  # 5488 x 1 x 1
                )
                 
                self.decoder = nn.Sequential(    
                    View([-1, 7, 28, 28]),
                    nn.ConvTranspose2d(7, 14, (3, 3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 32 x 64 x 64
                
                    nn.ConvTranspose2d(14, 56, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 64 x 128 x 128
        
                    nn.ConvTranspose2d(56, 112, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 64 x 128 x 128

                    nn.ConvTranspose2d(112, 224, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
                    nn.ReLU(True),
        
                    nn.ConvTranspose2d(224, 1, (3,3), stride=(1,1), padding=(1,1)),
                    nn.Sigmoid()
                )
        elif enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)
        else:
            self.encoder = valid_encoders[enc_type]["enc"](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]["dec"](self.latent_dim, self.input_height, first_conv, maxpool1)
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