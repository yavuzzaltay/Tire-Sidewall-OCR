# Docker ile Çalıştırma

Bu proje Docker ile paketlenmiştir; başka bir bilgisayarda Python/kütüphane kurmadan,
tek komutla web arayüzünü ayağa kaldırabilirsin.

## Gereksinim

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) veya
  Docker Engine + Compose (Linux)
- GPU **gerekmez** — imaj CPU üzerinde çalışacak şekilde hazırlandı (kod GPU varsa
  otomatik onu da kullanır, ama şart değil).

## Hızlı başlangıç

```bash
git clone https://github.com/yavuzzaltay/LastikOCR.git
cd LastikOCR

# Kendi lastik fotograflarini bu klasorlere koy:
#   Lastik_fotolari/Lastigin_ustu/Ust_1.jpg, Ust_2.jpg, ...
#   Lastik_fotolari/Lastigin_alti/Alt_1.jpg, Alt_2.jpg, ...

docker compose up --build
```

Sonra tarayıcıda **http://localhost:5000** adresini aç.

- Fotoğrafları seç (veya "Tümünü Seç"), "Seçilenleri İşle" butonuna bas.
- Adım adım ön işleme görselleri + çıkarılan alanlar + süre bilgisi ekranda görünür.
- Sonuçlar (`results/` klasörü) host makinede de kalır — container'ı durdurup
  kaldırsan bile kaybolmaz.

## Durdurma / yeniden başlatma

```bash
docker compose down        # durdur
docker compose up          # tekrar baslat (yeniden build gerekmez)
docker compose up --build  # kod degisti, yeniden derle
```

## Docker komutu olmadan (docker compose'suz)

```bash
docker build -t lastik-ocr .
docker run -p 5000:5000 \
  -v "$(pwd)/Lastik_fotolari:/app/Lastik_fotolari" \
  -v "$(pwd)/results:/app/results" \
  lastik-ocr
```

## Notlar

- İmaj **CPU-uyumlu** bağımlılıklarla kurulur (`requirements-docker.txt`) — GPU'ya özel
  `torch+cuda`/`onnxruntime-gpu` paketleri (~3-4GB) dahil değildir, bu yüzden imaj küçük
  ve her makinede sorunsuz build olur.
- Kod GPU'yu otomatik algılar (`src/engines/base.py:gpu_available()`). Eğer container'ı
  NVIDIA GPU'lu bir Linux makinede, `nvidia-container-toolkit` kurulu şekilde çalıştırırsan
  ve imajı GPU'lu `torch`/`onnxruntime-gpu` ile yeniden kurarsan (bkz. ana `requirements.txt`),
  otomatik olarak GPU'yu kullanır — ekstra kod değişikliği gerekmez.
- Web arayüzü varsayılan olarak hızlı tek-motor konfigürasyonu (`RapidOCR + v1_clahe`)
  kullanır — üretim raporundaki tam ensemble değil, görsel inceleme/keşif aracıdır.
- `Lastik_fotolari/` ve `results/` klasörleri `.gitignore`'da ve imaja dahil edilmez
  (`.dockerignore`) — fotoğraflarını her zaman kendi makinende, volume üzerinden sağlarsın.
