import typing as tp

from pm4py.objects.log import obj

from preprocessing import preprocessors


class BinaryTargetExtractor(preprocessors.BasePreprocessor):
    def __init__(self):
        super().__init__(key="target")

    def __call__(self, trace: obj.Trace, event_log: obj.EventLog) -> int:
        return int(trace.attributes["Accepted"])


class NextActivityTargetExtractor(preprocessors.BasePreprocessor):
    def __init__(self):
        super().__init__(key="target")
        self.activity2id: tp.Dict[str, int] = None

    def _calc_activity_mapping(self, event_log: obj.EventLog):
        unique_activities = set()
        for trace in event_log:
            for event in trace:
                unique_activities.add(event["concept:name"])

        self.activity2id = {
            # it's important to sort to calculate more easily node features
            activity: idx
            for idx, activity in enumerate(sorted(unique_activities))
        }
        self.id2activity = {idx: activity for activity, idx in self.activity2id.items()}

    def __call__(self, trace: obj.Trace, event_log: obj.EventLog) -> int:
        if self.activity2id is None:
            self._calc_activity_mapping(event_log)

        return self.activity2id[trace[-1]["concept:name"]]
