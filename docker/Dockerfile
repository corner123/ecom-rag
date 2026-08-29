FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv pip install --system --no-cache "pydantic>=2.9,<3" "pydantic-settings>=2.6,<3" "pytest>=8,<9"
COPY trade_agent ./trade_agent
COPY tests ./tests
CMD ["python", "-m", "trade_agent.cli"]
