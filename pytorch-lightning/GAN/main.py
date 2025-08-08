from model import GAN
from data import MNISTDataModule

import pytorch_lightning as pl

dm = MNISTDataModule()
model = GAN(*dm.dims)
trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=5,
)
trainer.fit(model, dm)