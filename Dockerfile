# syntax=docker/dockerfile:1.2
FROM python:3.11-slim
# put you docker configuration here

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY challenge/ ./challenge/
COPY data/trained_model.pkl ./data/trained_model.pkl

EXPOSE 8000

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000"]

