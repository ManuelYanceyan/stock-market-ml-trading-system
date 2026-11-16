# Container image for the ETF ML trading research pipeline.
#
# Build:
#   docker build -t etf-ml-trading .
# Run (downloads real data, trains, backtests, writes reports/):
#   docker run --rm -v "%cd%/reports:/app/reports" etf-ml-trading
#
# NOTE: this image is for research/simulation only. It never places orders.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages required by matplotlib (headless) and parquet tooling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Fail the build early if the code does not import.
RUN python -c "import src.run_pipeline, src.backtest, src.train_models"

# Default command: full live run (network needed for the data sources).
CMD ["python", "-m", "src.run_pipeline", "--mode", "live"]
