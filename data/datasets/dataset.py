import dataclasses
import typing as tp
from functools import partial, reduce
from uuid import uuid4

import numpy as np
import torch.utils.data as torch_data_utils
from pm4py.objects.log import obj
from torch_geometric import data as tg_data_utils
from tqdm import tqdm

from preprocessing import preprocessors as preprocessor_module
from utils.colors import ColorPrint


@dataclasses.dataclass
class DatasetItem:
    trace: obj.Trace
    label: int
    graph: tg_data_utils.Data


class Dataset(torch_data_utils.Dataset):
    preprocessed_data_key = "graph"
    target_key = "target"

    def __init__(
        self,
        event_log: obj.EventLog,
    ):
        self.event_log = event_log
        self.data = np.array(
            [{"trace_id": trace.attributes["concept:name"]} for trace in self.event_log]
        )

    def preprocess(
        self,
        preprocessors: tp.List[preprocessor_module.BasePreprocessor],
    ):
        for i, trace in tqdm(
            enumerate(self.event_log),
            total=len(self.event_log),
        ):
            for preprocessor in preprocessors:
                self.data[i][preprocessor.key] = preprocessor(
                    trace,
                    self.event_log,
                )

    def augment(self, transforms: list, augments_number: int = 1) -> None:
        graph_key = self.preprocessed_data_key
        # let's pretend for simplicity all nodes have same features
        augments: list[tp.Dict[str, tp.Any]] = [
            {
                **{"trace_id": str(uuid4())},
                **transforms[0](self.data, keys=[graph_key, self.target_key]),
            }
            for _ in tqdm(range(augments_number))
        ]

        print(
            ColorPrint.format(
                f"added {len(augments)} traces to {len(self.data)} existing",
                ColorPrint.WARNING,
            )
        )

        self.data = np.concatenate([self.data, np.array(augments)])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Any]:
        # TODO add augmentation
        return self.data[idx]
