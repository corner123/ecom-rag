FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app TRADE_AGENT_PACKAGED_SOURCE=1
RUN apt-get update && apt-get install -y --no-install-recommends git zsh \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv==0.9.21 && uv export --format requirements.txt --all-extras --locked > /tmp/requirements.txt && uv pip install --system --no-cache -r /tmp/requirements.txt
COPY .gitignore .env.example ./
COPY trade_agent ./trade_agent
COPY scripts ./scripts
COPY data ./data
COPY demo ./demo
COPY db ./db
COPY docs ./docs
COPY docker ./docker
COPY tests ./tests
COPY README.md ./README.md
COPY docker/healthcheck.py ./healthcheck.py
COPY docker-compose.yml ./docker-compose.yml
CMD ["python", "-m", "trade_agent.server"]
