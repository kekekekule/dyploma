import abc

import pm4py.objects.log.obj as data_utils


class BaseEventLoader(abc.ABC):
    @abc.abstractmethod
    def __call__(self, path_to_file: str) -> data_utils.EventLog:
        raise NotImplementedError
