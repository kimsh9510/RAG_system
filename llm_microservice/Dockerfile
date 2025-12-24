FROM python:3.12-slim

WORKDIR /app

# Install system build tools required for packages that run git/cmake/compile
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libomp-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install torch + matching nvidia packages first (already in your Dockerfile)
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py nodes.py ./
COPY llm_microservice/llm_service.py ./
COPY .env ./

EXPOSE 8080
CMD ["uvicorn", "llm_service:app", "--host", "0.0.0.0", "--port", "8080"]