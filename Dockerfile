FROM python:3.11-slim

# OpenCV 및 Pillow 필요한 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    fonts-noto-cjk \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Cloud Run은 PORT 환경변수를 사용하므로 하드코딩 OK
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
