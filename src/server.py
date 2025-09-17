from typing import cast

from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from rizemind.authentication.config import AccountConfig
from rizemind.authentication.eth_account_strategy import EthAccountStrategy
from rizemind.configuration.toml_config import TomlConfig
from rizemind.logging.metrics_storage_strategy import (
    MetricPhases,
    MetricsStorageStrategy,
)
from rizemind.logging.mlflow.config import MLFlowConfig
from rizemind.logging.mlflow.metrics_storage import MLFLowMetricStorage
from rizemind.strategies.contribution.shapley.decentralized.shapley_value_strategy import (
    DecentralShapleyValueStrategy,
)
from rizemind.strategies.contribution.shapley.shapley_value_strategy import (
    TrainerSetAggregate,
)
from rizemind.swarm.config import SwarmConfig
from rizemind.web3.config import Web3Config

from src.mlp import RealMLP
from src.task import EvaluationMetricSet, get_weights


def weighted_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
    total_num_examples = 0
    cum_rmse = 0.0
    cum_mae = 0.0
    cum_r2 = 0.0
    for num_examples, evaluation_metric_set in metrics:
        metric_set = EvaluationMetricSet.model_validate_json(
            cast(str, evaluation_metric_set["evaluation_metric_set"])
        )
        cum_rmse += metric_set.rmse * num_examples
        cum_mae += metric_set.mae * num_examples
        cum_r2 += metric_set.r2 * num_examples
        total_num_examples += num_examples
    # Aggregate and return custom metric (weighted average)
    return {
        "avg_rmse": cum_rmse / total_num_examples,
        "avg_mae": cum_mae / total_num_examples,
        "avg_r2": cum_r2 / total_num_examples,
    }


def weighted_aggregated_coalition_metrics(
    coalitions: list[TrainerSetAggregate],
) -> dict[str, Scalar]:
    cum_rmse: float = 0
    cum_mae: float = 0
    cum_r2: float = 0

    total_num_examples = 0
    for coalition in coalitions:
        evaluation = coalition._evaluation_res[0]
        total_num_examples += evaluation.num_examples
        metric_set = EvaluationMetricSet.model_validate_json(
            cast(str, evaluation.metrics.get("evaluation_metric_set"))
        )
        cum_rmse += metric_set.rmse * evaluation.num_examples
        cum_mae += metric_set.mae * evaluation.num_examples
        cum_r2 += metric_set.r2 * evaluation.num_examples
    return {
        "avg_rmse": cum_rmse / total_num_examples,
        "avg_mae": cum_mae / total_num_examples,
        "avg_r2": cum_r2 / total_num_examples,
    }


def on_fit_config(server_round: int):
    return {"lr_scaler": cast(Scalar, server_round)}


def server_fn(context: Context):
    ndarrays = get_weights(RealMLP(input_dim=8, output_dim=1))
    parameters = ndarrays_to_parameters(ndarrays)

    fedavg_strategy = FedAvg(
        fraction_fit=float(context.run_config["fraction-fit"]),
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_available_clients=int(context.run_config["min-available-clients"]),
        min_evaluate_clients=int(context.run_config["min-evaluate-clients"]),
        on_fit_config_fn=on_fit_config,
        evaluate_metrics_aggregation_fn=weighted_metrics,
        evaluate_fn=None,
        initial_parameters=parameters,
    )

    toml_config = TomlConfig("./pyproject.toml")
    auth_config = AccountConfig(**toml_config.get("tool.eth.account"))

    num_supernodes = int(context.run_config["num-supernodes"])

    account = auth_config.get_account(0)
    members = []
    for i in range(1, num_supernodes + 1):
        trainer = auth_config.get_account(i)
        members.append(trainer.address)

    web3_config = Web3Config(**toml_config.get("tool.web3"))
    w3 = web3_config.get_web3()
    swarm_config = SwarmConfig(**toml_config.get("tool.web3.swarm"))
    swarm = swarm_config.get_or_deploy(deployer=account, trainers=members, w3=w3)

    shapley_strategy = DecentralShapleyValueStrategy(
        fedavg_strategy,
        swarm,
        # Since we use loss for selecting the best model, we only provide
        # the base strategy and swarm, and leave the aggregate_coalition_metrics_fn
        # and coalition_to_score_fn as None.
        aggregate_coalition_metrics_fn=weighted_aggregated_coalition_metrics,
    )
    auth_strategy = EthAccountStrategy(shapley_strategy, swarm, account)

    # The order of strategies is important
    # The MLFLowMetricStrategy must come last
    # because DecentralShapleyValueStrategy relies on its superclass
    # implementation for aggregate_evaluate and configure_evaluate
    # which makes aggregate_evaluate and configure_evaluate in
    # MLFLowMetricStrategy unreachable.
    mlflow_config = MLFlowConfig(**toml_config.get("tool.mlflow.config"))
    mlflow_metrics_storage = MLFLowMetricStorage(
        experiment_name=mlflow_config.experiment_name,
        run_name=mlflow_config.run_name,
        mlflow_uri=mlflow_config.mlflow_uri,
    )
    metrics_storage_strategy = MetricsStorageStrategy(
        strategy=auth_strategy,
        metrics_storage=mlflow_metrics_storage,
        enabled_metric_phases=[MetricPhases.AGGREGATE_EVALUATE],
        save_best_model=True,
    )

    config = ServerConfig(num_rounds=int(context.run_config["num-server-rounds"]))
    return ServerAppComponents(strategy=metrics_storage_strategy, config=config)


app = ServerApp(server_fn=server_fn)
