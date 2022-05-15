import typing as tp
from re import L

import pytorch_lightning as pl
import torch
import torch_geometric.data as tg_data_utils
from pm4py.objects.log import obj

from augments.simple_augmentation import SimpleAugmentation
from data.datasets import dataset
from embeddings.base import BaseEmbedding
from preprocessing import preprocessors as preprocessors_module


class EventLogDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_event_log: obj.EventLog,
        val_event_log: obj.EventLog,
        test_event_log: obj.EventLog,
        batch_size: int,
        target_extractors: tp.List[preprocessors_module.BasePreprocessor],
        preprocessors: tp.Optional[tp.List[preprocessors_module.BasePreprocessor]],
    ):
        super().__init__()
        self.event_logs: tp.Dict[str, obj.EventLog] = {
            "train": train_event_log,
            "val": val_event_log,
            "test": test_event_log,
        }
        self.target_extractors = target_extractors
        self.preprocessors = preprocessors
        self.augmentation = SimpleAugmentation("aug")
        self.batch_size = batch_size
        self.datasets: tp.Dict[str, dataset.Dataset] = {}
        self.ready = False

    def download_data(self) -> None:
        raise NotImplementedError

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        self.datasets: tp.Dict[str, dataset.Dataset] = {
            key: dataset.Dataset(value) for key, value in self.event_logs.items()
        }
        if self.preprocessors is not None:
            for key in self.datasets:
                self.datasets[key].preprocess(self.preprocessors)
                # if key == "train":
                #     self.datasets["train"].augment([self.augmentation], 2000)

        for key in self.datasets:
            self.datasets[key].preprocess(self.target_extractors)

        self.ready = True

    def train_dataloader(self):
        return tg_data_utils.DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return tg_data_utils.DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return tg_data_utils.DataLoader(
            self.datasets["test"], batch_size=self.batch_size
        )
