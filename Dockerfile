FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# Build the model into the image: fetch, clean, select and calibrate. This
# makes the container self-contained so the API has a model to serve on boot.
RUN python -m src.train

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
