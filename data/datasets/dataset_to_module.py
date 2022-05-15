from data.datasets.bpi2012.event_order_extractor import (
    EventOrderExtractor as BPI2012EventOrderExtractor,
)
from data.datasets.bpi2012.loader import EventLoader as BPI2012Loader
from data.datasets.bpi2012.target_extractors import (
    NextActivityTargetExtractor as BPI2012TargetExtractor,
)
from data.datasets.bpi2012w.event_order_extractor import (
    EventOrderExtractor as BPI2012wEventOrderExtractor,
)
from data.datasets.bpi2012w.loader import EventLoader as BPI2012wLoader
from data.datasets.bpi2012w.target_extractors import (
    NextActivityTargetExtractor as BPI2012wTargetExtractor,
)
from data.datasets.bpi2017.event_order_extractor import (
    EventOrderExtractor as BPI2017EventOrderExtractor,
)
from data.datasets.bpi2017.loader import EventLoader as BPI2017Loader
from data.datasets.bpi2017.target_extractors import (
    NextActivityTargetExtractor as BPI2017TargetExtractor,
)
from data.datasets.bpi2017w.event_order_extractor import (
    EventOrderExtractor as BPI2017wEventOrderExtractor,
)
from data.datasets.bpi2017w.loader import EventLoader as BPI2017wLoader
from data.datasets.bpi2017w.target_extractors import (
    NextActivityTargetExtractor as BPI2017wTargetExtractor,
)
from data.datasets.helpdesk.event_order_extractor import (
    EventOrderExtractor as HelpdeskEventOrderExtractor,
)
from data.datasets.helpdesk.loader import EventLoader as HelpdeskLoader
from data.datasets.helpdesk.target_extractors import (
    NextActivityTargetExtractor as HelpdeskTargetExtractor,
)

DATASET_TO_MODULE = {
    "bpi2017w": {
        "loader": BPI2017wLoader,
        "target_extractor": BPI2017wTargetExtractor,
        "order_extractor": BPI2017wEventOrderExtractor,
    },
    "bpi2012w": {
        "loader": BPI2012wLoader,
        "target_extractor": BPI2012wTargetExtractor,
        "order_extractor": BPI2012wEventOrderExtractor,
    },
    "helpdesk": {
        "loader": HelpdeskLoader,
        "target_extractor": HelpdeskTargetExtractor,
        "order_extractor": HelpdeskEventOrderExtractor,
    },
    "bpi2017": {
        "loader": BPI2017Loader,
        "target_extractor": BPI2017TargetExtractor,
        "order_extractor": BPI2017EventOrderExtractor,
    },
    "bpi2012": {
        "loader": BPI2012Loader,
        "target_extractor": BPI2012TargetExtractor,
        "order_extractor": BPI2012EventOrderExtractor,
    },
}
