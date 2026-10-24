FROM python:3.12-slim
WORKDIR /srv
# libgomp is required by xgboost / lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY dengue_features.py ./
COPY models ./models
COPY app ./app
ENV PORT=8080
EXPOSE 8080
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --timeout 60 app.app:app
