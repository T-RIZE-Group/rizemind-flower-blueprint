# Rizemind + Flower Blueprint

A minimal blueprint showing how to integrate `Rizemind` with `Flower` for federated learning. It trains a simple `RealMLP` on the California Housing dataset using Flower clients.

## Features

- **Federated training**: Flower server with multiple clients
- **Dataset**: California Housing via `flwr-datasets`
- **Model**: `RealMLP` (PyTorch)
- **Metrics**: Loss, RMSE, MAE, R²
- **RizeMind**: For Verifiability, transparency, accountability, decentralization, and robustness
- **MLflow**: run/experiment tracking
- **Docker**: compose setup for supernode/superlink, plus `superexec` image

## Requirements

- Python 3.12+
- uv <https://docs.astral.sh/uv/>
- Docker and Docker Compose

## Install dependencies

```bash
uv sync --all-groups
```

## Generate and configure credentials

1. Generate an account mnemonic (via `rzmnd` CLI):

   ```bash
   uv run rzmnd account generate NAME_OF_YOUR_ACCOUNT
   ```

2. Copy the generated mnemonic file to your local keystore:

   - Place it at `.rzmnd/keystore/rizenet_testnet_example.json` (or your chosen file name - make sure to change the `superexec.Dockerfile` if necessary)

3. Fill the following sections in `pyproject.toml`:

   ```toml
   [tool.eth.account.mnemonic_store]
   account_name = "account_name"
   passphrase  = "passphrase"

   [tool.mlflow.config]
   run_name        = "run_name"
   experiment_name = "experiment_name"
   mlflow_uri      = "mlflow_url"

   ```

## Build Docker image(s)

Build the `superexec` image used by the deployment:

```bash
docker build -f docker/superexec.Dockerfile -t rizemind_superexec:0.0.1 .
```

### Create network

```bash
docker network create flwr-network
```

### Start services

Bring up Superlink and Supernode via compose (in two steps):

```bash
docker compose -f docker/superlink.docker-compose.yaml up -d

docker compose -f docker/supernode.docker-compose.yaml up -d
```

### Run the Flower deployment

With the services running, start the Flower app (local deployment):

```bash
uv run flwr run . local-deployment --stream
```

This will launch the server and clients as configured and stream logs to your terminal.

### Tear down

```bash
docker compose -f docker/supernode.docker-compose.yaml down

docker compose -f docker/superlink.docker-compose.yaml down

docker network rm flwr-network
```
