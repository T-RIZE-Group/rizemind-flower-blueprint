FROM flwr/superexec:latest

WORKDIR /app

COPY --chown=app:app pyproject.toml README.md .python-version uv.lock ./
RUN python -m pip install uv \
    && uv sync


ENTRYPOINT ["uv", "run", "--", "flower-superexec"]