FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "uvicorn[standard]"
# Fastapi port
EXPOSE 9000

CMD ["uvicorn", "semantic_proxy:app", "--host", "0.0.0.0", "--port", "9000"]
