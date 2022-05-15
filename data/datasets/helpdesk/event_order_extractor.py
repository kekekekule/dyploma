from pm4py.objects.log import obj

from preprocessing import preprocessors


class EventOrderExtractor(preprocessors.BasePreprocessor):
    def __init__(self):
        super().__init__(key="events_order")

    def __call__(self, trace: obj.Trace, event_log: obj.EventLog) -> int:
        return [activity["concept:name"] for activity in trace]
