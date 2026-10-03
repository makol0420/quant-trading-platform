# Runs the dashboard API (uvicorn api.main:app) only. The trading loop
# (scripts/run_paper_trade.py or a live equivalent) is intentionally a
# separate process/container -- see README.md "Going live safely" for why
# "the dashboard is running" and "the bot is trading" should never share
# one on/off switch.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

EXPOSE 8000

# start.sh builds the gitignored artifacts (cached OHLCV, models, backtest
# results, report) in the background, then execs uvicorn. Running
# `uvicorn api.main:app` directly still works, but the dashboard will have
# nothing real to show -- see README "Deploying".
CMD ["./start.sh"]
