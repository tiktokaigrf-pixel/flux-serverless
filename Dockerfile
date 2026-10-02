# vast.ai Base-Image: enthält CUDA, Caddy, Auth-Tools
FROM vastai/pytorch:latest

WORKDIR /app

# System-Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python-Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Code
COPY server.py .
COPY start-server.sh .
RUN chmod +x start-server.sh

# Umgebungsvariablen
ENV MODEL_NAME="black-forest-labs/FLUX.1-schnell"
ENV PYTHONUNBUFFERED=1

# Port für den Inference-Server
EXPOSE 18000

# Startup
CMD ["/app/start-server.sh"]
