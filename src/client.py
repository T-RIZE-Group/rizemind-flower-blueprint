import torch
from eth_account import Account
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays
from polars import DataFrame
from rizemind.authentication.authentication_mod import authentication_mod
from rizemind.authentication.config import ACCOUNT_CONFIG_STATE_KEY, AccountConfig
from rizemind.authentication.notary.model.mod import model_notary_mod
from rizemind.configuration.toml_config import TomlConfig
from rizemind.logging.mlflow.config import MLFLOW_CONFIG_KEY, MLFlowConfig
from rizemind.logging.mlflow.mod import mlflow_mod
from rizemind.strategies.contribution.shapley.decentralized.shapley_value_client import (
    DecentralShapleyValueClient,
)
from rizemind.web3.config import WEB3_CONFIG_STATE_KEY, Web3Config

from src.mlp import RealMLP
from src.task import get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        X_train: DataFrame,
        y_train: DataFrame,
        X_val: DataFrame,
        y_val: DataFrame,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        self.model = RealMLP(input_dim=input_dim, output_dim=output_dim)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(
        self, parameters: NDArrays, config: dict[str, bool | bytes | float | int | str]
    ) -> tuple[NDArrays, int, dict[str, bool | bytes | float | int | str]]:
        set_weights(model=self.model, parameters=parameters)
        state_dict, metric_history = train(
            model=self.model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            epochs=self.epochs,
            device_type=self.device_type,
            batch_size=self.batch_size,
            base_lr=self.learning_rate,
            lr_scaler=float(config["lr_scaler"]) if "lr_scaler" in config else 1,
        )
        self.model.load_state_dict(state_dict=state_dict, strict=True)

        return (
            get_weights(self.model),
            len(self.X_train),
            {
                **metric_history.serialize(),
            },
        )

    def evaluate(
        self, parameters: NDArrays, config: dict[str, bool | bytes | float | int | str]
    ) -> tuple[float, int, dict[str, bool | bytes | float | int | str]]:
        set_weights(self.model, parameters)
        metric_set = test(
            model=self.model,
            X_test=self.X_val,
            y_test=self.y_val,
            device_type=self.device_type,
        )
        return (
            metric_set.loss,
            len(self.X_val),
            {"evaluation_metric_set": metric_set.model_dump_json()},
        )


def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])

    X_train, y_train, X_val, y_val = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
    )

    toml_config = TomlConfig("./pyproject.toml")
    account_config = AccountConfig(
        **toml_config.get("tool.eth.account")
        | {"default_account_index": partition_id + 1}
    )
    context.state.config_records[ACCOUNT_CONFIG_STATE_KEY] = (
        account_config.to_config_record()
    )
    web3_config = Web3Config(**toml_config.get("tool.web3"))
    context.state.config_records[WEB3_CONFIG_STATE_KEY] = web3_config.to_config_record()

    mlflow_config = MLFlowConfig(**toml_config.get("tool.mlflow.config"))
    context.state.config_records[MLFLOW_CONFIG_KEY] = mlflow_config.to_config_record()

    flwr_client = FlowerClient(
        input_dim=8,
        output_dim=1,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    shapley_client = DecentralShapleyValueClient(flwr_client)
    return shapley_client.to_client()


Account.enable_unaudited_hdwallet_features()
app = ClientApp(
    client_fn=client_fn,
    mods=[
        mlflow_mod,
        authentication_mod,
        model_notary_mod,
    ],
)
