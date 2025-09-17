FROM flwr/superexec:latest

WORKDIR /app

COPY --chown=app:app pyproject.toml README.md .python-version uv.lock ./
RUN python -m pip install uv \
    && uv sync

COPY --chown=app:app .rzmnd/keystore/rizenet_testnet_example.json .rzmnd/keystore/rizenet_testnet_example.json

ENTRYPOINT ["uv", "run", "--", "flower-superexec"]