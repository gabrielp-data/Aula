FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# deps primeiro (cache separado do código)
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project

# código
COPY src/ ./src/
RUN uv sync --frozen

# o dado NÃO é copiado: vem via dvc pull ou bind mount em dev
CMD ["uv", "run", "python", "-m", "churn.model"]
