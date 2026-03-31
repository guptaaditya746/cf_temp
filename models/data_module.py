import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class UnsupervisedTimeSeriesDataset(Dataset):
    def __init__(self, data_array):
        self.data = torch.tensor(data_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AtacamaDataModule(pl.LightningDataModule):
    def __init__(self, x_train, x_val, x_calib, x_test, batch_size=64, num_workers=2):
        super().__init__()
        self.x_train = x_train
        self.x_val = x_val
        self.x_calib = x_calib
        self.x_test = x_test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = UnsupervisedTimeSeriesDataset(self.x_train)
            self.val_dataset = UnsupervisedTimeSeriesDataset(self.x_val)

        if stage == "test" or stage is None:
            self.test_dataset = UnsupervisedTimeSeriesDataset(self.x_test)

        if stage == "predict" or stage is None:
            self.calib_dataset = UnsupervisedTimeSeriesDataset(self.x_calib)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.calib_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
