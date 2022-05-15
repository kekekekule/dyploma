import dataclasses
import json
import typing as tp
from argparse import ArgumentParser
from pprint import pprint

import pytorch_lightning as pl

import wandb
from data import event_log_data_module
from data.datasets import base_event_loader
from data.datasets.dataset_to_module import DATASET_TO_MODULE
from models.slug_to_class import SLUG_TO_CLASS as MODEL_SLUG_TO_CLASS
from preprocessing import preprocessors as preprocessors_module
from preprocessing.preprocessor_to_module import PREPROCESSOR_SLUG_TO_CLASS
from utils.colors import ColorPrint as colors
from utils.helpers import merge_dicts_recursive, unflatten_dict
from utils.wrappers import JobWithMessage


def _build_preprocessor(
    preprocessor_slug: str,
    **kwargs,
) -> preprocessors_module.BasePreprocessor:
    class_obj = PREPROCESSOR_SLUG_TO_CLASS[preprocessor_slug]
    return class_obj(preprocessor_slug, **kwargs)


def build_preprocessors(
    preprocessor_slugs: tp.List[str],
    **kwargs,
) -> tp.Optional[tp.List[preprocessors_module.BasePreprocessor]]:
    if not preprocessor_slugs:
        return None
    return [_build_preprocessor(slug, **kwargs) for slug in preprocessor_slugs]


def build_data_module(
    dataset_name: str,
    path_to_train: str,
    path_to_val: str,
    path_to_test: str,
    batch_size: int,
    preprocessors: tp.Optional[tp.List[preprocessors_module.BasePreprocessor]],
):
    loader_class = DATASET_TO_MODULE[dataset_name]["loader"]
    target_extractor_class = DATASET_TO_MODULE[dataset_name]["target_extractor"]
    event_loader: base_event_loader.BaseEventLoader = loader_class()
    target_extractor: preprocessors_module.BasePreprocessor = target_extractor_class()

    train_events = event_loader(path_to_train)
    val_events = event_loader(path_to_val)
    test_events = event_loader(path_to_test)

    return event_log_data_module.EventLogDataModule(
        train_event_log=train_events,
        val_event_log=val_events,
        test_event_log=test_events,
        batch_size=batch_size,
        target_extractors=[target_extractor],
        preprocessors=preprocessors,
    )


def build_model(
    model_slug,
    model_config: tp.Dict[str, tp.Any],
):
    model_class = MODEL_SLUG_TO_CLASS[model_slug]
    model = model_class.from_configuration(model_config)
    return model


@dataclasses.dataclass
class Config:
    dataset: str
    model_slug: str
    path_to_train: str
    path_to_val: str
    path_to_test: str
    batch_size: int
    preprocessor_slugs: tp.List[str]
    model_config: tp.Dict[str, tp.Any]
    train_config: tp.Dict[str, tp.Any]
    run_name: str
    embeddings: tp.Dict[str, tp.Any]

    def to_dict(self):
        return {field: getattr(self, field) for field in self.__annotations__}

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            dataset=config["dataset"],
            model_slug=config["model_slug"],
            path_to_train=config["path_to_train"],
            path_to_val=config["path_to_val"],
            path_to_test=config["path_to_test"],
            batch_size=config["batch_size"],
            preprocessor_slugs=config["preprocessor_slugs"],
            model_config=config["model_config"],
            train_config=config["train_config"],
            run_name=config["run_name"],
            embeddings=config["embeddings"],
        )


parser = ArgumentParser()
parser.add_argument("--path_to_config", default=None)
parser.add_argument("--sweep", default=None)
parser.add_argument("--reuse", action='store_true')
args = parser.parse_args()


data_module = None


def main():
    global data_module

    with JobWithMessage("Loading config..."):
        with open(args.path_to_config, "r") as fin:
            base_config = unflatten_dict(json.load(fin))
        with open(args.sweep, "r") as fin:
            sweep_config = unflatten_dict(json.load(fin))
            sweep_defaults = {
                key: sweep_config["parameters"][key].pop("default")
                for key in sweep_config["parameters"]
            }

    with JobWithMessage("Loading wandb..."):
        logger = pl.loggers.WandbLogger(
            project="gnn4pbpm", name=base_config["run_name"]
        )
        logger.experiment.config.update(sweep_defaults)
        current_step_sweep_config = unflatten_dict(wandb.config)
        config = Config.from_config(
            merge_dicts_recursive(base_config, current_step_sweep_config)
        )

    print(colors.format("Here is the config:", colors.WARNING))
    pprint(config.to_dict())

    with JobWithMessage("Building data..."):
        if data_module is None or not args.reuse:
            preprocessors = build_preprocessors(
                preprocessor_slugs=config.preprocessor_slugs,
                **{"embeddings": config.embeddings},
            )
            data_module = build_data_module(
                dataset_name=config.dataset,
                path_to_train=config.path_to_train,
                path_to_val=config.path_to_val,
                path_to_test=config.path_to_test,
                batch_size=config.batch_size,
                preprocessors=preprocessors,
            )

    with JobWithMessage("Setup model..."):
        if not data_module.ready or not args.reuse:
            data_module.setup()
        model = build_model(
            model_slug=config.model_slug,
            model_config=config.model_config,
        )
        trainer = pl.Trainer(
            logger=logger, deterministic=True, accelerator="auto", **config.train_config
        )

    print(colors.format("Ready to train!", colors.OKCYAN))
    trainer.fit(model, data_module)
    trainer.test(dataloaders=data_module.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    with open(args.sweep, "r") as fin:
        sweep_config = unflatten_dict(json.load(fin))
        sweep_defaults = {
            key: sweep_config["parameters"][key].pop("default")
            for key in sweep_config["parameters"]
        }
        sweep_id = wandb.sweep(sweep_config, project="gnn4pbpm")
        print(colors.format(f"sweep_id: {sweep_id}", colors.WARNING))
    wandb.agent(sweep_id, main)
    main()
