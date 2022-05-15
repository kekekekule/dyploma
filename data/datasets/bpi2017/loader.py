import uuid
from copy import deepcopy

import pandas as pd
import pm4py
import pm4py.objects.log.obj as data_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.util import xes_constants as xes
from tqdm import tqdm

from .. import base_event_loader


class EventLoader(base_event_loader.BaseEventLoader):
    def __call__(self, path_to_file: str) -> data_utils.EventLog:
        df = pd.read_csv(path_to_file, sep=",")
        event_log = self.filter_and_cast_to_pm4py_format(df)
        return event_log

    @staticmethod
    def filter_and_cast_to_pm4py_format(df: pd.DataFrame) -> data_utils.EventLog:
        log_csv = dataframe_utils.convert_timestamp_columns_in_df(df)
        log_csv.rename(
            columns={"Accepted": "case:Accepted", "Activity": "concept:name"},
            inplace=True,
        )
        for group_name, df_group in tqdm(
            log_csv.groupby("case:concept:name"),
            total=log_csv["case:concept:name"].nunique(),
        ):
            prefix = []
            for row_index, row in df_group.sort_values("time:timestamp").iterrows():
                row_dict = row.to_dict()
                prefix.append(data_utils.Event(row_dict, attributes={"origin": "csv"}))
                prefix_uuid = str(uuid.uuid4())
                for record in prefix:
                    trace_id = str(record["case:concept:name"]) + prefix_uuid
                    record["case:concept:name"] = trace_id
                log_list.append(
                    data_utils.Trace(
                        deepcopy(prefix), attributes={xes.DEFAULT_TRACEID_KEY: trace_id}
                    )
                )
                for record in prefix:
                    record["case:concept:name"] = record["case:concept:name"].replace(
                        prefix_uuid, ""
                    )
        del log_csv
        print("Loading new DataFrame, length is", len(log_list))
        # log_csv = pd.DataFrame(log_list)
        # log_csv = log_csv.sort_values("time:timestamp")
        print("Converting... [stream]")
        event_stream = data_utils.EventStream(log_list)
        del log_list
        print("Converting... [apply]")
        event_log = data_utils.EventLog(event_stream)
        # event_log = log_converter.apply(
        #     event_stream,
        #     parameters={
        #         log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"
        #     },
        # )
        print("Filtering...")
        event_log = pm4py.filter_log(lambda trace: len(trace) > 2, event_log)
        return event_log
