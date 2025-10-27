FROM python:3.10.6-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8080"]


