import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pyformulas as pf

from .utils import ZiCDataset
import matplotlib.pyplot as plt


class VAEBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """
        self.save_hyperparameters(hparams)

        self.screen = pf.screen(title = "generated glyph")

    def setup(self, stage):
        pass

    def train_dataloader(self):
        self.trainset = ZiCDataset(self.hparams, stage = "train", device = "cpu")
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=self.hparams["batch_size"], num_workers=1, shuffle = True)
        else:
            return None

    def val_dataloader(self):
        self.valset = ZiCDataset(self.hparams, stage = "val", device = "cpu")
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=self.hparams["batch_size"], num_workers=1)
        else:
            return None

    def test_dataloader(self):
        self.testset = ZiCDataset(self.testset, self.hparams, stage = "test", device = "cpu")
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=self.hparams["batch_size"], num_workers=1)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    def encode(self, input, idxs):

        raise NotImplementedError

    def decode(self, z, idxs):

        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_loss_weights(self, out_imgs):
        positive_pixels = (out_imgs > 0).sum()
        negative_pixels = (out_imgs <= 0).sum()
        weights = torch.empty_like(out_imgs)
        weights[out_imgs > 0] = 0.5/(positive_pixels + 1e-12)
        weights[out_imgs <= 0] = 0.5/(negative_pixels + 1e-12)
        return weights
    
    def training_step(self, batch, batch_idx):
        
        outputs, mu, log_var = self(batch["in_imgs"], batch["in_idxs"], batch["out_idxs"])

        recons_loss = ((outputs - batch["out_imgs"]).square() * self.get_loss_weights(batch["out_imgs"])).sum()
        kld_loss = -0.5*((1 + log_var - mu ** 2 - log_var.exp()).sum(-1)).mean()
        sigmoid = lambda x: 1/(np.exp(-x)+1)
        loss = sigmoid(self.hparams["log_weight_ratio"])*recons_loss +  sigmoid(-self.hparams["log_weight_ratio"])*kld_loss
        if self.hparams["plot"]:
            self.screen.update(torch.cat([batch["out_imgs"][0], outputs[0]], dim = -1).squeeze().unsqueeze(-1).expand(-1, -1, 3).detach().cpu().numpy())
        
        self.log_dict(
            {
                "training_loss": loss, 
                "reconstruction_loss": recons_loss,
                "KL_divergence_loss": kld_loss
            }
        )

        return loss


    def shared_evaluation(self, batch, batch_idx, log=False):

        outputs, mu, logvar = self(batch["in_imgs"], batch["in_idxs"], batch["out_idxs"])

        recons_loss = F.mse_loss(outputs, batch["out_imgs"])
        kld_loss = -0.5*((1 + logvar - mu ** 2 - logvar.exp()).sum(-1)).mean()
        sigmoid = lambda x: 1/(np.exp(-x)+1)
        loss = sigmoid(self.hparams["log_weight_ratio"])*recons_loss +  sigmoid(-self.hparams["log_weight_ratio"])*kld_loss

        if self.hparams["plot"]:
            self.screen.update(torch.cat([batch["out_imgs"][0], outputs[0]], dim = -1).squeeze().unsqueeze(-1).expand(-1, -1, 3).cpu().numpy())
            
        self.log_dict(
            {
                "val_loss": loss, 
                "val_reconstruction_loss": recons_loss,
                "val_KL_divergence_loss": kld_loss
            }
        )

        return outputs, loss

    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs[1]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs[1]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()