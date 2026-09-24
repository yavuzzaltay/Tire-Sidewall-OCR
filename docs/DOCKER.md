# Docker ile Çalıştırma

Bu proje Docker ile paketlenmiştir; başka bir bilgisayarda Python/kütüphane kurmadan,
tek komutla web arayüzünü ayağa kaldırabilirsin.

## Gereksinim

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) veya
  Docker Engine + Compose (Linux)
- GPU **gerekmez** — varsayılan imaj CPU üzerinde çalışacak şekilde hazırlandı. NVIDIA
  GPU'lu bir makinede daha hızlı çalıştırmak istersen aşağıdaki "GPU ile çalıştırma"
  bölümüne bak (ayrı, test edilmiş bir GPU imajı var).

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

## GPU ile çalıştırma (NVIDIA GPU'lu makinede)

Ayrı bir GPU imajı var (`Dockerfile.gpu` + `docker-compose.gpu.yml`) — **gerçek donanımda test edildi**
(RTX 3060: CPU'da 99.5 sn süren tek lastik işlemi GPU'da 29.7 sn'ye indi, `onnxruntime` ve
`torch` tarafında CUDA doğrulandı).

### GPU'lu makinede gereken kurulum

1. **NVIDIA GPU sürücüsü** (güncel, WSL2/CUDA destekli) — çoğu makinede zaten kurulu.
2. **Docker Desktop** — kurulumda "Use WSL 2 based engine" seçili olmalı (varsayılan).
   Docker Desktop 4.x sürümleri WSL2 üzerinde NVIDIA GPU'yu otomatik tanır, ekstra bir
   "NVIDIA Container Toolkit" kurulumu **Windows'ta gerekmez** (Linux'ta gerekir, aşağıya bak).
3. Kurulumdan sonra doğrulama: `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi`
   komutu GPU bilgisini basıyorsa hazırsın.

### Çalıştırma

```bash
git clone https://github.com/yavuzzaltay/LastikOCR.git
cd LastikOCR
# Kendi fotograflarini Lastik_fotolari/Lastigin_ustu ve Lastigin_alti klasorlerine koy

docker compose -f docker-compose.gpu.yml up --build
```

İlk build biraz uzun sürer (CUDA taban imajı + torch/onnxruntime-gpu indirmesi, ~10-15dk,
imaj boyutu ~15GB) — sonraki çalıştırmalarda `--build` gerekmez, direkt
`docker compose -f docker-compose.gpu.yml up`.

Loglarda `"GPU kullanılıyor"` yazısını görmelisin. Görmüyorsan (CPU'ya düşmüşse):

```bash
docker exec <container_adi> python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

çıktısında `CUDAExecutionProvider` yoksa GPU geçişi (device passthrough) çalışmıyor demektir —
Docker Desktop ayarlarında GPU desteğinin açık olduğunu kontrol et.

### Linux'ta (Docker Desktop değil, düz Docker Engine)

Ek olarak [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
kurman gerekir (`nvidia-ctk runtime configure --runtime=docker` + `sudo systemctl restart docker`).

## Notlar

- **CPU imajı** (`Dockerfile`, varsayılan `docker compose up`) **CPU-uyumlu** bağımlılıklarla
   kurulur (`requirements-docker.txt`) — GPU'ya özel paketler dahil değildir, ~3GB,
   GPU olmayan makinelerde de sorunsuz build olur.
- **GPU imajı** (`Dockerfile.gpu`, `docker-compose.gpu.yml`) NVIDIA'nın CUDA+cuDNN taban
  imajını kullanır, ~15GB. Kod GPU'yu otomatik algılar (`src/engines/base.py:gpu_available()`),
  ekstra ayar gerekmez — sadece doğru compose dosyasıyla (`-f docker-compose.gpu.yml`) build et.
- Web arayüzü varsayılan olarak hızlı tek-motor konfigürasyonu (`RapidOCR + v1_clahe`)
  kullanır — üretim raporundaki tam ensemble değil, görsel inceleme/keşif aracıdır.
- `Lastik_fotolari/` ve `results/` klasörleri `.gitignore`'da ve imaja dahil edilmez
  (`.dockerignore`) — fotoğraflarını her zaman kendi makinende, volume üzerinden sağlarsın.
