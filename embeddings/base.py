import abc
import typing as tp

from pm4py.objects.log import obj


class BaseEmbedding(abc.ABC):
    def __init__(self, key: str):
        self.key = key

    @abc.abstractmethod
    def __call__(self, event_log: obj.EventLog) -> tp.Any:
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, entity: tp.Any):
        raise NotImplementedError
