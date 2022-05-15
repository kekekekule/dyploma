import random
import typing as tp
from collections import defaultdict

import numpy as np
import torch
from torch_geometric import data as tg_data_utils
from tqdm import tqdm

from preprocessing.preprocessors import BasePreprocessor


class SimpleAugmentation(BasePreprocessor):
    TERMINAL_ID = "__TERM"
    START_ID = "__START"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ready = False

    def _dfs(self, vertex: tp.Any, path: list = None, used: dict = None):
        if vertex == self.TERMINAL_ID:
            if len(path) > 2:
                return path
            return []

        used[vertex] = True

        neighbors = list(self.graph[vertex])[:]
        random.shuffle(neighbors)
        for neigh in neighbors:
            if not used.get(neigh, False):
                path.append(neigh)
                result = self._dfs(neigh, path, used)
                if result:
                    return_value = (
                        result[:-1] if result[-1] == self.TERMINAL_ID else result
                    )
                    # print(return_value)
                    return return_value
                path.pop()

        return None

    def _build(self, data):
        self.graph = defaultdict(set)

        for trace in tqdm(data):
            edge_index = trace["graph"]["edge_index"].T
            for i, (fr, to) in enumerate(edge_index):
                self.graph[int(fr)].add(int(to))

                if not i:
                    self.graph[self.START_ID].add(int(fr))

                if i + 1 == len(edge_index):
                    self.graph[int(to)].add(self.TERMINAL_ID)

        self.ready = True

    def _check_readiness_and_build(self, data):
        if self.ready:
            return

        self._build(data)

    def _iterate_path(self, path):
        previous = path[0]
        for i in range(1, len(path)):
            yield previous, path[i]
            previous = path[i]

    def get_graph(self, data: tg_data_utils):
        self._check_readiness_and_build(data)
        path = self._dfs(self.START_ID, [], {})
        # print(path)
        assert len(path) > 1

        edge_index = torch.tensor(
            np.array([[fr, to] for fr, to in self._iterate_path(path)]).T,
            dtype=torch.long,
        )

        return tg_data_utils.Data(edge_index=edge_index, x=data[0]["graph"]["x"])

    def get_target(self, _: tg_data_utils):
        return [self.graph__temporary["edge_index"][0][0]] + list(
            self.graph__temporary["edge_index"][1, :]
        )

    def __call__(self, data, keys=[]):
        result = {}
        for key in keys:
            result[key] = getattr(self, f"get_{key}")(data)
            setattr(self, f"{key}__temporary", result[key])
        return result
