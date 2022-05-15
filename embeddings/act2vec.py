import os
import pathlib
import typing as tp
from uuid import uuid4

import gensim
import numpy as np
import torch
from pm4py.objects.log import obj

from embeddings.base import BaseEmbedding

PATH = pathlib.Path(__file__).parent.resolve()


class Act2Vec(BaseEmbedding):
    def __init__(
        self,
        key: str,
        vectorsize: int,
        start_alpha: float,
        learning_rate: float,
        n_epochs: int,
        window: int,
        dataset_name: str,
        cache: tp.Optional[str],
        save: bool,
        **kwargs,
    ):
        super().__init__(key)
        self.vectorsize = vectorsize
        self.start_alpha = start_alpha
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.window = window
        self.dataset_name = dataset_name
        self.cache = cache
        self.model = None
        self.save = save

    @classmethod
    def from_configuration(cls, configuration: tp.Dict[str, tp.Any]):
        return cls(**configuration)

    def _extract_per_trace_activities(self, event_log: obj.EventLog):
        return [[event[self.key] for event in trace] for trace in event_log]

    def _init_and_train(self, event_log: obj.EventLog):
        sentences = self._extract_per_trace_activities(event_log)
        size_kw = "size"
        if "vector_size" in gensim.models.Word2Vec.__init__.__code__.co_varnames:
            size_kw = "vector_size"  # for backward compat

        if self.cache:
            cache_dirname = os.path.join(PATH, f"{self.__class__.__name__}_cache")
            print(f"Loading act2vec from {self.cache}")
            self.model = gensim.models.Word2Vec.load(
                os.path.join(cache_dirname, self.cache)
            )
        else:
            self.model = gensim.models.Word2Vec(
                sentences,
                window=self.window,
                min_count=0,
                **{size_kw: self.vectorsize},
            )
            for epoch in range(self.n_epochs):
                print(f"[{self.__class__.__name__}] Now training epoch {epoch}")
                self.model.train(
                    sentences,
                    start_alpha=self.start_alpha,
                    epochs=self.n_epochs,
                    total_examples=self.model.corpus_count,
                )
                self.model.alpha -= self.learning_rate
                self.model.min_alpha = self.model.alpha

        if self.save:
            cache_dirname = os.path.join(PATH, f"{self.__class__.__name__}_cache")
            try:
                os.mkdir(cache_dirname)
            except Exception:  # already exists
                ...

            cache_filename = f"{self.dataset_name}_{str(uuid4())}.model"
            full_path = os.path.join(cache_dirname, cache_filename)
            print(f"Saving model as {cache_filename} to {full_path}")

            self.model.save(full_path)

        return self

    def _get_activity2vec_dict(self, activities: tp.Union[dict, list]):
        act2vec = {}
        for event in activities:
            if event not in act2vec:
                act2vec[event] = self.model.wv[event]
        return act2vec

    def __call__(self, event_log: obj.EventLog) -> tp.Any:
        return self._init_and_train(event_log)

    def apply(self, activities: tp.Union[dict, list]) -> torch.Tensor:
        assert self.model
        activity2vec = self._get_activity2vec_dict(activities)

        all_unique_activities = sorted(set(activities))

        return torch.Tensor(
            np.array([activity2vec[event_name] for event_name in all_unique_activities])
        )
