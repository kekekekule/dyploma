import abc
from collections import defaultdict
import typing as tp

import numpy as np
import torch
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.objects.log import obj
from pm4py.util import xes_constants as xes_util
from torch_geometric import data as tg_data_utils

from embeddings.slug_to_class import EMBEDDINGS_SLUG_TO_CLASS
from preprocessing import preprocessors
from utils.colors import ColorPrint as colors

activity_key = xes_util.DEFAULT_NAME_KEY
timestamp_key = xes_util.DEFAULT_TIMESTAMP_KEY


class BaseGraphBuilder(preprocessors.BasePreprocessor):
    @abc.abstractmethod
    def __call__(self, trace: obj.Trace, event_log: obj.EventLog) -> tg_data_utils.Data:
        raise NotImplementedError


class DFGGraphBuilder(BaseGraphBuilder):
    embeddings = None

    def __init__(self, key: str, **kwargs):
        super().__init__(key="graph")

        self.activity2id: tp.Optional[tp.Dict[str, int]] = None
        self.id2activity: tp.Optional[tp.Dict[int, str]] = None

        if "embeddings" in kwargs and kwargs["embeddings"] is not None:
            slug = kwargs["embeddings"].pop("slug")
            self.embeddings = EMBEDDINGS_SLUG_TO_CLASS[slug].from_configuration(
                {
                    "key": xes_util.DEFAULT_NAME_KEY,
                    **kwargs["embeddings"],
                }
            )

    @staticmethod
    def _iter_activity_pairs(trace: obj.Trace):
        trace_iterator = iter(trace)
        previous_activity = next(trace_iterator)[activity_key]

        for _ in range(len(trace) - 1):
            next_activity = next(trace_iterator)[activity_key]
            yield previous_activity, next_activity
            previous_activity = next_activity

    def build_tg_graph(
        self,
        trace: obj.Trace,
        node_features: tp.Optional[torch.Tensor] = None,
    ):
        # print('node features size: ', node_features)
        edges: tp.List[tp.Tuple[int, int]] = []
        ordered_activities = []
        assert len(trace) > 0
        for fr, to in self._iter_activity_pairs(trace):
            fr_vertex_id = self.activity2id[fr]
            to_vertex_id = self.activity2id[to]
            edges.append((fr_vertex_id, to_vertex_id))
            ordered_activities.append(fr)
            ordered_activities.append(to)

        edges.pop()  # pop last edge, that we need to predict
        mapping = node_features if node_features is not None else self.activity2id
        edge_index = torch.tensor(edges, dtype=torch.long).T

        if node_features is None:
            ordered_activities = sorted(set(self.activity2id))
            activity_id_features = np.array(
                [mapping[act] for act in ordered_activities]
            )
            assert len(activity_id_features) > 0
            node_features = torch.tensor(activity_id_features, dtype=torch.float)

        assert len(node_features) > 0
        assert len(edge_index) > 0
        return tg_data_utils.Data(
            x=node_features, edge_index=edge_index, id2activity=self.id2activity, counts=self.counts_by_id,
        )

    def _calc_activity_mapping(self, event_log: obj.EventLog):
        unique_activities = set()
        counts_by_activity = defaultdict(int)
        for trace in event_log:
            for event in trace:
                unique_activities.add(event["concept:name"])
                counts_by_activity[event["concept:name"]] += 1

        self.activity2id = {
            # it's important to sort to calculate more easily node features
            str(activity): idx
            for idx, activity in enumerate(sorted(unique_activities))
        }

        acts_sum = sum(counts_by_activity.values())

        self.counts_by_id = {
            self.activity2id[act_]: [count_ / acts_sum]
            for act_, count_ in counts_by_activity.items()
        }
        self.id2activity = {idx: activity for activity, idx in self.activity2id.items()}

    def _prepare_node_features(
        self, event_log: obj.EventLog
    ) -> tp.Dict[str, tp.Union[list, torch.Tensor]]:
        if self.embeddings is None:
            return None

        if self.embeddings.model is None:
            self.embeddings = self.embeddings(event_log)

        return self.embeddings.apply(self.activity2id)

    def __call__(self, trace: obj.Trace, event_log: obj.EventLog) -> tg_data_utils.Data:
        if self.activity2id is None:
            self._calc_activity_mapping(event_log)
        assert len(trace) > 1
        node_features: torch.Tensor = self._prepare_node_features(event_log)

        return self.build_tg_graph(trace, node_features=node_features)
