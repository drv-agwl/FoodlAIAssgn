from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from torch import optim
import numpy as np

class LitGenerator(pl.LightningModule):
    def __init__(self, model, mode="classification"):
        super().__init__()
        self.model = model
        self.train_draw_period = 100
        self.val_draw = False
        self.mode = mode

    def training_step(self, batch, batch_idx):
        rgb, ir, smell, y = batch['rgb_image'].permute(0, 3, 1, 2), batch["ir_image"].permute(0, 3, 1, 2), batch["smell"], batch["label"]
        z = self.model(rgb, ir, smell)

        if self.mode=="classification":
            loss = torch.nn.CrossEntropyLoss()(z, y.long().reshape(-1,))
        else:
            loss = F.mse_loss(z, y)

        self.log("train_loss", loss)

        if self.mode=="classification":
            preds = z.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            acc = np.sum(preds == y.detach().cpu().numpy().reshape(-1, )) / z.shape[0]

            self.log("train_acc", acc*100.)

        return loss

    def validation_step(self, batch, batch_idx):
        rgb, ir, smell, y = batch['rgb_image'].permute(0, 3, 1, 2), batch["ir_image"].permute(0, 3, 1, 2), batch["smell"], batch["label"]
        z = self.model(rgb, ir, smell)

        if self.mode == "classification":
            loss = torch.nn.CrossEntropyLoss()(z, y.long().reshape(-1,))
        else:
            loss = F.mse_loss(z, y)

        self.log("val_loss", loss)

        if self.mode=="classification":
            preds = z.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            acc = np.sum(preds == y.detach().cpu().numpy().reshape(-1,)) / z.shape[0]

            self.log("val_acc", acc*100.)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # sched = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-2, epochs=1000, steps_per_epoch=110,
        #                                       pct_start=0.3, anneal_strategy='cos')
        #
        # return {"optimizer": optimizer,
        #         "lr_scheduler": sched}

        return optimizer