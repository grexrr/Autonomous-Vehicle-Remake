FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codes
COPY . .

# Port
EXPOSE 5000

# Run
CMD ["gunicorn", "--config", "gunicorn_config.py", "api.wsgi:app"]