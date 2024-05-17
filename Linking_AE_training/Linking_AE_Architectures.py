from pytorch_lightning import LightningModule
from torch import nn
import torch
from torch.nn import functional as F

class Linking_AE(LightningModule):
    def __init__(
        self,
        lr: float = 1e-5,
        input_layers: int = 4,
        hidden_layer: int = 768,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr

        if input_layers == 4:
            self.encoder = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(True),
                nn.Linear(768, hidden_layer),
            )

            self.decoder = nn.Sequential(          
                nn.Linear(hidden_layer, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 4096),
            )
        elif input_layers == 3:
            self.encoder = nn.Sequential(
                nn.Linear(768, hidden_layer),
            )

            self.decoder = nn.Sequential(          
                nn.Linear(hidden_layer, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 4096),
            )
        elif input_layers == 5:
            self.encoder = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(True),
                nn.Linear(768, hidden_layer),
            )

            self.decoder = nn.Sequential(          
                nn.Linear(hidden_layer, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 4096),
            )
        elif input_layers == 6:
            self.encoder = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(True),
                nn.Linear(768, hidden_layer),
            )

            self.decoder = nn.Sequential(          
                nn.Linear(hidden_layer, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 4096),
            )
        elif input_layers == 7:
            self.encoder = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(True),
                nn.Linear(768, 768),
                nn.ReLU(True),
                nn.Linear(768, hidden_layer),
            )

            self.decoder = nn.Sequential(          
                nn.Linear(hidden_layer, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 4096),
            )
        elif input_layers == 8:
            self.encoder = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(True),
                nn.Linear(768, 768),
                nn.ReLU(True),
                nn.Linear(768, 768),
                nn.ReLU(True),
                nn.Linear(768, hidden_layer),
            )

            self.decoder = nn.Sequential(          
                nn.Linear(hidden_layer, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 4096),
            )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def _run_step(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def step(self, batch, batch_idx):
        x, y = batch
        x_hat = self._run_step(x)
        loss = F.mse_loss(x_hat, y)
        logs = {
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
