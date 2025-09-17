import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, cast

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torchmetrics
from flwr.common import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from pydantic import BaseModel
from rizemind.logging.train_metric_history import TrainMetricHistory
from torch.utils.data import DataLoader, TensorDataset

from src.mlp import RealMLP

warnings.filterwarnings(action="ignore", module="flwr_datasets")


class EvaluationMetricSet(BaseModel):
    loss: float
    rmse: float
    mae: float
    r2: float

    model_config = {"frozen": True}


def _calculate_metrics(
    y_pred: torch.Tensor, y_target: torch.Tensor
) -> EvaluationMetricSet:
    y_pred, y_target = y_pred.cpu(), y_target.cpu()
    mse_loss: float = nn.functional.mse_loss(y_pred, y_target).item()
    rmse_metric: float = np.sqrt(mse_loss)
    mae_metric: float = torchmetrics.functional.mean_absolute_error(
        y_pred, y_target
    ).item()
    r2_metric: float = torchmetrics.functional.r2_score(y_pred, y_target).item()

    return EvaluationMetricSet(
        loss=mse_loss, rmse=rmse_metric, mae=mae_metric, r2=r2_metric
    )


def train(
    *,
    model: RealMLP,
    X_train: pl.DataFrame,
    y_train: pl.DataFrame,
    X_val: pl.DataFrame,
    y_val: pl.DataFrame,
    epochs: int,
    device_type: str,
    batch_size: int = 256,
    base_lr: float = 0.07,
    lr_scaler: float = 1,
) -> tuple[dict[str, Any], TrainMetricHistory]:
    device = torch.device(device=device_type)
    model = model.to(device)

    ds_train = TensorDataset(
        X_train.to_torch(dtype=pl.Float32), y_train.to_torch(dtype=pl.Float32)
    )
    dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)

    params: list[nn.Parameter] = list(model.parameters())
    scale_params: list[nn.Parameter] = [params[0]]
    weight_params: list[nn.Parameter] = params[1::2]
    bias_params: list[nn.Parameter] = params[2::2]

    optimizer = torch.optim.Adam(
        params=[
            dict(params=scale_params),
            dict(params=weight_params),
            dict(params=bias_params),
        ],
        betas=(0.9, 0.95),
    )
    criterion = nn.MSELoss()

    n_train_batches: int = len(dl_train)
    best_val_loss: float = np.inf

    best_state: dict[str, Any] = {}
    metrics = TrainMetricHistory()

    for _ in range(epochs):
        model.train()
        train_loss: float = 0.0

        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []

        for batch_idx, (x_batch, y_batch) in enumerate(dl_train):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Cosine-on-log schedule - similar to original RealMLP paper but instead of epochs
            # we rely on lr_scaler which is server round
            # 100 == num-server-rounds
            t = ((lr_scaler * n_train_batches) + batch_idx) / (100 * n_train_batches)
            lr_sched: float = 0.5 - 0.5 * np.cos(2 * np.pi * np.log2(1 + 15 * t))
            lr: float = base_lr * lr_sched
            optimizer.param_groups[0]["lr"] = 6 * lr  # scale
            optimizer.param_groups[1]["lr"] = 1 * lr  # weights
            optimizer.param_groups[2]["lr"] = 0.1 * lr  # biases

            optimizer.zero_grad()

            # forward and backward pass
            y_pred: torch.Tensor = model.forward(x_batch)
            loss: torch.Tensor = criterion.forward(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            predictions.append(y_pred)
            targets.append(y_batch)

        train_pred_full = torch.concat(predictions, dim=0)
        train_target_full = torch.concat(targets, dim=0)

        train_metrics = _calculate_metrics(train_pred_full, train_target_full)

        metrics.append(metrics=train_metrics.model_dump(), is_eval=False)

        model.eval()
        current_val_metrics = test(
            model=model, X_test=X_val, y_test=y_val, device_type=device_type
        )
        metrics.append(current_val_metrics.model_dump(), is_eval=True)

        val_loss = current_val_metrics.loss
        if val_loss <= best_val_loss:  # Use <= to prefer later epochs with same loss
            best_val_loss = val_loss
            best_state = deepcopy(model.cpu().state_dict())

    return best_state, metrics


def test(
    *,
    model: RealMLP,
    X_test: pl.DataFrame,
    y_test: pl.DataFrame,
    device_type: str,
    eval_batch_size: int = 512,
) -> EvaluationMetricSet:
    device = torch.device(device_type)
    model = model.to(device)

    ds_test = TensorDataset(
        X_test.to_torch(dtype=pl.Float32), y_test.to_torch(dtype=pl.Float32)
    )
    dl_test = DataLoader(ds_test, batch_size=eval_batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []

        for x_batch, y_batch in dl_test:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model.forward(x_batch)
            predictions.append(y_pred)
            targets.append(y_batch)

        test_pred_full = torch.concat(predictions, dim=0)
        test_target_full = torch.concat(targets, dim=0)
        metrics = _calculate_metrics(test_pred_full, test_target_full)

    return metrics


def get_weights(model: nn.Module) -> list[Any]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, parameters: NDArrays):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


fds: FederatedDataset | None = None  # Cache FederatedDataset


def load_data(
    partition_id: int, num_partitions: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    global fds
    if fds is None:
        train_partitioner = IidPartitioner(num_partitions=num_partitions)
        eval_partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="gvlassis/california_housing",
            partitioners={"train": train_partitioner, "validation": eval_partitioner},
        )

    ds_train: pl.DataFrame = cast(
        pl.DataFrame,
        fds.load_partition(partition_id=partition_id, split="train").with_format(
            type="polars"
        )[:],
    )
    X_train: pl.DataFrame = ds_train.drop("MedHouseVal")
    y_train: pl.DataFrame = ds_train.select("MedHouseVal")

    ds_eval: pl.DataFrame = cast(
        pl.DataFrame,
        fds.load_partition(partition_id=partition_id, split="validation").with_format(
            type="polars"
        )[:],
    )
    X_eval: pl.DataFrame = ds_eval.drop("MedHouseVal")
    y_eval: pl.DataFrame = ds_eval.select("MedHouseVal")

    return X_train, y_train, X_eval, y_eval
