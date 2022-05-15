import typing as tp

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric.nn as tg_nn
import torch_scatter
import torchmetrics
from torch import optim
from torch.nn import functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

import wandb
from layers.attention import GlobalAttention as NodeAttention

def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class AttentiveGGNN(pl.LightningModule):
    def __init__(
        self,
        num_activities: int,
        num_timestamps: int,
        hidden_size: int,
        optimizer_config: dict,
        use_standard_embedding: tp.Optional[int],
    ):
        super().__init__()

        self.num_activities = num_activities
        self.hidden_size = hidden_size

        self.use_standard_embedding = use_standard_embedding

        self.embeddings = None
        if use_standard_embedding:
            self.embeddings = nn.Embedding(num_activities, use_standard_embedding)

        self.gcn1 = tg_nn.GatedGraphConv(hidden_size, num_timestamps)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.gcn2 = tg_nn.GatedGraphConv(hidden_size, num_timestamps)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.gcn3 = tg_nn.GatedGraphConv(hidden_size, num_timestamps)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.gcn4 = tg_nn.GatedGraphConv(hidden_size, num_timestamps)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)

        self.attn = NodeAttention(nn.Linear(hidden_size, 1))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=hidden_size // 2, out_features=hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=hidden_size // 4, out_features=num_activities),
            nn.LogSoftmax(dim=-1),
        )

        self._precision = torchmetrics.Precision(
            num_classes=num_activities, average="macro"
        )
        self.val_precision = self._precision.clone()
        self.recall = torchmetrics.Recall(num_classes=num_activities, average="macro")
        self.val_recall = self.recall.clone()
        self.accuracy = torchmetrics.Accuracy(num_classes=num_activities)
        self.val_accuracy = self.accuracy.clone()
        self.f1_score = torchmetrics.F1Score(
            num_classes=num_activities, average="macro"
        )
        self.val_f1_score = self.f1_score.clone()

        self.test_accuracy = self.accuracy.clone()
        self.test_f1_score = self.f1_score.clone()

        for module in self.classifier:
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                print(f"Assigning xavier uniform to {module.__class__.__name__}")
                torch.nn.init.xavier_uniform_(module.weight)

        self.accumulated_scores_one_epoch = torch.zeros(num_activities)
        self.accumulated_scores = torch.zeros(num_activities)
        self.id2activity = None
        self.optimizer_config = optimizer_config

    @classmethod
    def from_configuration(cls, configuration: tp.Dict[str, tp.Any]):
        return cls(
            num_activities=configuration["num_activities"],
            num_timestamps=configuration["num_timestamps"],
            hidden_size=configuration["hidden_size"],
            optimizer_config=configuration["optimizer"],
            use_standard_embedding=configuration["use_standard_embedding"],
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_config["learning_rate"])
        scheduler_first = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.optimizer_config["step_size"],
            gamma=self.optimizer_config["gamma"],
            verbose=self.optimizer_config["verbose"],
        )
        return [optimizer], [scheduler_first]

    def forward(self, node_embeddings, edge_index, batch_idx):
        if self.embeddings:
            node_embeddings = self.embeddings(node_embeddings.long())

        node_embeddings1: torch.Tensor = (
            F.relu(self.gcn1(node_embeddings, edge_index))
        )
        # node_embeddings1 = torch.cat([gmp(node_embeddings, batch_idx), gap(node_embeddings1, batch_idx)], dim=1)
        node_embeddings2 = (F.relu(self.gcn2(node_embeddings1, edge_index)))
        # node_embeddings2 = torch.cat([gmp(node_embeddings2, batch_idx)], dim=1)
        node_embeddings3 = (F.relu(self.gcn3(node_embeddings2, edge_index)))
        # node_embeddings3 = torch.cat([gmp(node_embeddings3, batch_idx)], dim=1)
        node_embeddings4 = self.bn4(F.relu(self.gcn4(node_embeddings3, edge_index)))
        # node_embeddings_pooled = torch_scatter.scatter_mean(
        #     node_embeddings4,
        #     torch.tensor(list(range(self.num_activities)) * (node_embeddings4.size(0) // self.num_activities), dtype=torch.int64),
        #     dim=0
        # )
        node_embeddings4, attn = self.attn(node_embeddings4, batch_idx)
        if self.training:
            with torch.no_grad():
                idx = torch.tensor(
                    list(range(self.num_activities))
                    * (node_embeddings.size(0) // self.num_activities),
                    dtype=torch.int64,
                )
                self.accumulated_scores_one_epoch += torch_scatter.scatter_mean(
                    attn, idx, dim=0
                ).squeeze(-1)
        node_embeddings4 = torch.cat([gmp(node_embeddings4, batch_idx)], dim=1)

        # print(node_embeddings_pooled.size())
        logits = self.classifier(node_embeddings4)
        return logits  # .squeeze(-1)

    def _run(self, batch: tp.Dict[str, tp.Any]):
        graphs = batch["graph"]

        if self.id2activity is None:
            # я такие костыли сами знаете где видел, но ничего не поделаешь
            self.id2activity = graphs[0]["id2activity"]

        logits: torch.Tensor = self(graphs["x"], graphs["edge_index"], graphs.batch)
        # print(logits)
        # print(logits.size())
        avg_loss = F.cross_entropy(logits, batch["target"].long())

        self.log("train/loss_step", avg_loss, on_step=True)
        return {
            "loss": avg_loss,
            "logits": logits,
            "targets": batch["target"],
        }

    def training_step(self, batch: tp.Dict[str, tp.Any], batch_idx: int):
        return self._run(batch)

    def on_train_start(self):
        pl.seed_everything(42, workers=True)

    def training_step_end(self, outs):
        preds = torch.argmax(outs["logits"], axis=1)
        self.accuracy.update(preds.long(), outs["targets"].long())
        self.f1_score.update(preds.long(), outs["targets"].long())
        self._precision.update(preds.long(), outs["targets"].long())
        self.recall.update(preds.long(), outs["targets"].long())

    def training_epoch_end(self, outputs):
        self.accuracy.compute()
        self.f1_score.compute()
        self._precision.compute()
        self.recall.compute()

        self.accumulated_scores = (
            self.accumulated_scores_one_epoch
            / torch.sum(self.accumulated_scores_one_epoch).item()
        )
        self.accumulated_scores_one_epoch = torch.zeros(self.num_activities)
        table = wandb.Table(
            data=[
                (self.id2activity[column], value.item())
                for column, value in enumerate(self.accumulated_scores)
            ],
            columns=["activity", "importance"],
        )
        wandb.log(
            {
                "activity_importance": wandb.plot.bar(
                    table, "activity", "importance", title="Activities impact"
                )
            }
        )

        # print(self.accumulated_scores)
        # print(torch.argsort(self.accumulated_scores, descending=True))

    def validation_step(self, batch: tp.Dict[str, tp.Any], batch_idx: int):
        return self._run(batch)

    def test_step(self, batch: tp.Dict[str, tp.Any], batch_idx: int):
        return self._run(batch)

    def test_step_end(self, outs):
        preds = torch.argmax(outs["logits"], axis=1)
        self.test_accuracy.update(preds.long(), outs["targets"].long())
        self.test_f1_score.update(preds.long(), outs["targets"].long())

    def test_epoch_end(self, outs):
        self.log("test/accuracy", self.test_accuracy.compute())
        self.log("test/f1_score", self.test_f1_score.compute())

    def validation_step_end(self, outs):
        preds = torch.argmax(outs["logits"], axis=1)
        self.val_accuracy.update(preds.long(), outs["targets"].long())
        self.val_f1_score.update(preds.long(), outs["targets"].long())
        self.val_precision.update(preds.long(), outs["targets"].long())
        self.val_recall.update(preds.long(), outs["targets"].long())

    def validation_epoch_end(self, outs):
        self.log("validation/accuracy", self.val_accuracy.compute())
        self.log("validation/f1_score", self.val_f1_score.compute())
        self.log("validation/precision", self.val_precision.compute())
        self.log("validation/recall", self.val_recall.compute())
