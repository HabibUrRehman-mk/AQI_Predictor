FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY backend/models ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]