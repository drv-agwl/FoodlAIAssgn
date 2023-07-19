import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os.path as osp


def normalize(x):
    return 2*x - 1

class FoodAIDataset(Dataset):
    def __init__(self, split, rgb_timestamps, ir_timestamps, smell_data, num_classes,
                 rgb_data_dir="./pasra/pasra/0/rgb/", ir_data_dir="./pasra/pasra/0/ir", mode="classification"):
        super().__init__()
        self.rgb_timestamps = rgb_timestamps
        self.ir_timestamps = ir_timestamps
        self.smell_data = smell_data
        self.num_classes = num_classes
        self.rgb_data_dir = rgb_data_dir
        self.ir_data_dir = ir_data_dir

        self.rgb_timestamps = np.array(self.rgb_timestamps).reshape(num_classes, -1)
        self.ir_timestamps = np.array(self.ir_timestamps).reshape(num_classes, -1)
        self.smell_data = self.smell_data.reshape(num_classes, -1, self.smell_data.shape[-1])

        self.per_class_size = self.rgb_timestamps.shape[-1]

        if mode == "classification":
            self.labels = np.array([[x]*self.per_class_size for x in range(self.num_classes)])
        else:
            self.labels = np.array([x for x in range(self.per_class_size*self.num_classes)]).astype(np.float32) / (self.per_class_size * self.num_classes)
            self.labels = self.labels.reshape(num_classes, -1)

        if split=="train":
            self.rgb_timestamps = self.rgb_timestamps[:, :int(0.8*self.per_class_size)]
            self.ir_timestamps = self.ir_timestamps[:, :int(0.8 * self.per_class_size)]
            self.smell_data = self.smell_data[:, :int(0.8 * self.per_class_size), :]
            self.labels = self.labels[:, :int(0.8 * self.per_class_size)]
        elif split=="val":
            self.rgb_timestamps = self.rgb_timestamps[:, int(0.8 * self.per_class_size):int(0.9 * self.per_class_size)]
            self.ir_timestamps = self.ir_timestamps[:, int(0.8 * self.per_class_size):int(0.9 * self.per_class_size)]
            self.smell_data = self.smell_data[:, int(0.8 * self.per_class_size):int(0.9 * self.per_class_size), :]
            self.labels = self.labels[:, int(0.8 * self.per_class_size):int(0.9 * self.per_class_size)]
        else:
            self.rgb_timestamps = self.rgb_timestamps[:, int(0.9 * self.per_class_size):]
            self.ir_timestamps = self.ir_timestamps[:, int(0.9 * self.per_class_size):]
            self.smell_data = self.smell_data[:, int(0.9 * self.per_class_size):, :]
            self.labels = self.labels[:, int(0.9 * self.per_class_size):]

        self.rgb_timestamps = self.rgb_timestamps.reshape(-1)
        self.ir_timestamps = self.ir_timestamps.reshape(-1)
        self.smell_data = self.smell_data.reshape(-1, self.smell_data.shape[-1])
        self.labels = self.labels.reshape(-1)

    def __len__(self):
        return len(self.rgb_timestamps)

    def __getitem__(self, idx):
        rgb_image_name = self.rgb_timestamps[idx]
        rgb_image = np.array(Image.open(osp.join(self.rgb_data_dir, rgb_image_name)))/255.
        ir_image_name = self.ir_timestamps[idx]
        ir_image = np.array(Image.open(osp.join(self.ir_data_dir, ir_image_name))) / 255.

        smell_data = self.smell_data[idx] # assuming already preprocessed

        rgb_image = normalize(rgb_image) # rescale to [-1, 1]
        ir_image = normalize(ir_image)

        return {"rgb_image": torch.Tensor(rgb_image), "ir_image": torch.Tensor(ir_image),
                "smell": torch.Tensor(smell_data), "label": torch.Tensor(self.labels[idx].reshape(1,))}



def get_dataloader(
    split,
    rgb_timestamps, ir_timestamps, smell_data, num_classes, rgb_data_dir="./pasra/pasra/0/rgb", ir_data_dir="./pasra/pasra/0/ir",
    training=True,
    num_workers=0,
    batch_size=64,
    mode="classification"
) -> DataLoader:

    data = FoodAIDataset(split, rgb_timestamps, ir_timestamps, smell_data, num_classes, rgb_data_dir, ir_data_dir, mode=mode)

    return DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        drop_last=training,
        pin_memory=True,
    )