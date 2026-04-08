FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY env /app/env
COPY server /app/server
COPY inference.py /app/inference.py
COPY openenv.yaml /app/openenv.yaml
COPY README.md /app/README.md

CMD ["python", "-m", "server.app"]
