# Lastik Yanak OCR — web arayuzu (CPU). Baska bir makinede GPU olmasa da calisir;
# kod GPU'yu otomatik algilar (src/engines/base.py:gpu_available()).
FROM python:3.11-slim

# opencv (libGL/libglib) ve Tesseract icin sistem bagimliliklari
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/
COPY data/ ./data/

# Lastik fotograflari ve sonuclar disaridan volume olarak baglanir (bkz. docker-compose.yml);
# imaj icinde bos klasorler olusturmak container'in bunlar olmadan da ayaga kalkmasini saglar.
RUN mkdir -p Lastik_fotolari/Lastigin_ustu Lastik_fotolari/Lastigin_alti results

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

CMD ["python", "src/webapp.py"]
