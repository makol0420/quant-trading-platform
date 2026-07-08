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

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
