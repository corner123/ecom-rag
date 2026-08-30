FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv export --format requirements.txt --all-extras --locked > /tmp/requirements.txt && uv pip install --system --no-cache -r /tmp/requirements.txt
COPY trade_agent ./trade_agent
COPY db ./db
COPY tests ./tests
COPY docker/healthcheck.py ./healthcheck.py
COPY docker/mysql.Dockerfile ./docker/mysql.Dockerfile
COPY docker-compose.yml ./docker-compose.yml
CMD ["python", "-m", "trade_agent.server"]
