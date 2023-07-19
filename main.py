from train import LitGenerator
from models.cnn import SimpleCNNWithVector
import os
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import datetime
from utils import find_closest_string
from data import FoodAIDataset, get_dataloader
import os
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

if __name__ == '__main__':
    data_dir_rgb = "./pasra/pasra/0/rgb"
    data_dir_ir = "./pasra/pasra/0/ir"
    mode="classification"

    smell_data = pd.read_csv("./pasra/pasra/0/nose/nose.csv")

    rgb_timestamps = [(float(x.split('.')[0]), x) for x in os.listdir(data_dir_rgb)]
    ir_timestamps = [(float(x.split('.')[0]), x) for x in os.listdir(data_dir_ir)]
    nose_timestamps = list(smell_data.iloc[:-1, -1])

    rgb_filtered_timestamps = [find_closest_string(rgb_timestamps, x) for x in nose_timestamps]
    ir_filtered_timestamps = [find_closest_string(ir_timestamps, x) for x in nose_timestamps]
    nose_data = np.array(smell_data.iloc[:-1, :-1])

    train_loader = get_dataloader("train", rgb_filtered_timestamps, ir_filtered_timestamps, nose_data, num_classes=8, training=True,
                                  num_workers=0, batch_size=32, mode=mode)
    val_loader = get_dataloader("val", rgb_filtered_timestamps, ir_filtered_timestamps, nose_data, num_classes=8, training=False,
                                  num_workers=0, batch_size=32, mode=mode)

    model = SimpleCNNWithVector(num_classes=8, input_vector_dim=4, mode=mode)
    lit_model = LitGenerator(model=model)

    tb_logger = pl_loggers.TensorBoardLogger(version="classification", save_dir='./')
    trainer = pl.Trainer(max_epochs=1000, accelerator="cuda",
                         log_every_n_steps=25, logger=tb_logger)
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)