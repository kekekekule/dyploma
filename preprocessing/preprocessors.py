import abc
import typing as tp

from pm4py.objects.log import obj


class BasePreprocessor(abc.ABC):
    def __init__(self, key: str):
        self.key = key

    @abc.abstractmethod
    def __call__(self, trace: obj.Trace, event_log: obj.EventLog) -> tp.Any:
        raise NotImplementedError
