# Web demo (Gradio) in a container.
#   docker build -t asl-demo .
#   docker run -p 7860:7860 asl-demo
# Put the models you want to serve in models/ (TFLite files in models/tflite/ keep the image small).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt
COPY src ./src
COPY app ./app
COPY models ./models
ENV PORT=7860
EXPOSE 7860
CMD ["python", "app/gradio_app.py"]
