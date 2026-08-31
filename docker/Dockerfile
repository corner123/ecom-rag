FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv==0.9.21 && uv export --format requirements.txt --all-extras --locked > /tmp/requirements.txt && uv pip install --system --no-cache -r /tmp/requirements.txt
COPY trade_agent ./trade_agent
COPY scripts ./scripts
COPY data ./data
COPY demo ./demo
COPY db ./db
COPY tests ./tests
COPY README.md ./README.md
COPY docker/healthcheck.py ./healthcheck.py
COPY docker/mysql.Dockerfile ./docker/mysql.Dockerfile
COPY docker-compose.yml ./docker-compose.yml
CMD ["python", "-m", "trade_agent.server"]
