import uuid
from copy import deepcopy

import pandas as pd
import pm4py
import pm4py.objects.log.obj as data_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
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
            columns={"Activity": "concept:name"},
            inplace=True,
        )
        print(log_csv["concept:name"].nunique(), "activities")
        log_list = []

        activities_to_exclude = {
            None,
            # "Resolve SW anomaly",
            # "Schedule intervention",
            # "VERIFIED",
            # "RESOLVED",
            # "INVALID",
            # "DUPLICATE",
        }

        cases_to_exclude = []

        for group_name, df_group in tqdm(
            log_csv.groupby("Case ID"), total=log_csv["Case ID"].nunique()
        ):
            if (set(df_group["concept:name"].unique()) & activities_to_exclude):
                cases_to_exclude.append(group_name)
        print(f"Exclude {len(activities_to_exclude)} activities")
        log_csv = log_csv[~log_csv["Case ID"].isin(cases_to_exclude)]

        for group_name, df_group in tqdm(
            log_csv.groupby("Case ID"), total=log_csv["Case ID"].nunique()
        ):
            prefix = []
            for row_index, row in df_group.iterrows():
                row_dict = row.to_dict()
                prefix.append(row_dict)
                prefix_uuid = str(uuid.uuid4())
                for record in prefix:
                    record["Case ID"] += prefix_uuid
                log_list.extend(deepcopy(prefix))
                for record in prefix:
                    record["Case ID"] = record["Case ID"].replace(prefix_uuid, "")
        log_csv = pd.DataFrame(log_list)
        log_csv = log_csv.sort_values("Complete Timestamp")
        event_log = log_converter.apply(
            log_csv,
            parameters={
                log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "Case ID"
            },
        )
        event_log = pm4py.filter_log(lambda trace: len(trace) > 2, event_log)
        print(log_csv["concept:name"].nunique(), "activities")
        return event_log
