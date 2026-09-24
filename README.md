# Lastik Yanak OCR — Tire Sidewall OCR

Line-scan kamerayla çekilmiş lastik çevresi görüntülerinden (**düşük kontrastlı kabartma yazı**)
yapısal bilgi çıkaran, **tamamen yerel/offline** çalışan uçtan uca bir OCR hattı + görsel web arayüzü.

Staj kapsamında geliştirildi: 119 lastik (238 görüntü) üzerinde doğrulandı, motor/varyant benchmark'ları
ve fizibilite raporuyla birlikte teslim edildi.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OCR](https://img.shields.io/badge/OCR-RapidOCR%20%7C%20EasyOCR%20%7C%20PP--OCRv6-green)
![Docker](https://img.shields.io/badge/Docker-CPU%20%2B%20GPU-blue)
![Offline](https://img.shields.io/badge/offline-tamamen%20yerel-lightgrey)

---

## Neden iyi bir proje?

- **Gerçek bir problemi çözüyor:** Kabartma lastik yazısı; klasik OCR'ın en zorlandığı alanlardan biri
  (düşük kontrast, ışık/gölge bağımlılığı, kavisli yüzey). Genel amaçlı OCR'ı doğrudan sürmek **%0 sonuç**
  veriyor — bu proje o sorunu mühendislikle aşıyor.
- **Ölçüme dayalı mühendislik:** 3 OCR motoru × 5 ön işleme varyantı etiketli sette karşılaştırıldı,
  kazananlar veriye göre seçildi. İki tur Ar-Ge sonunda doğruluk **%56.3 → %63.2** (30 lastiklik sette),
  alternatif konfigürasyonda **%71.8**.
- **Gerçek hız kazanımı:** GPU taşıma sonrası RapidOCR **3.1×**, EasyOCR **6.4×** hızlandı; iki motorlu
  ensemble CPU'daki tek motordan bile **daha hızlı** hale geldi (19.0 sn/lastik).
- **Tek komutla demo:** Docker imajı sayesinde kurulum derdi yok — `docker compose up` ile web arayüzü
  açılır, fotoğraf seçilir, adım adım sonuç görülür.
- **Üretime hazır düşünülmüş:** Kalite kapısı (boş çekim eleme), güven skorları, "emin değilsem sus"
  politikası, modüler motor/varyant mimarisi, detaylı rapor ve sonraki-adım önerileri dahil.

---

## Çalışırken — gerçek çıktılar

Aşağıdaki görseller pipeline'ın gerçek bir lastik (`Ust_5.jpg`, Continental EcoContact 6) üzerindeki
adım adım çıktısıdır. Sistem bu lastikte **8 kritik alanın 7'sini doğru** okudu
(ebat, hız grubu, marka, desen, mevsim, DOT, tüpsüz).

### 1. Ham görüntü → metin bandı tespiti

| Ham görüntü (temsili kesit, tam boyut 1116×12900 px) | Tespit edilen metin bandı (622 px) |
|---|---|
| ![Ham görüntü](assets/screenshots/01-ham-goruntu.png) | ![Bant tespiti](assets/screenshots/02-orientasyon-bant.png) |

Otomatik orientasyon (iki 180° adayın OCR-güven skoruyla oylanması) + satır parlaklık profiliyle
arka plan kırpma. Boş/başarısız çekimler kalite kapısında eleniyor (11/119 üst görüntüde doğru çalıştı).

### 2. Kontrast artırma — 5 varyant

![5 ön işleme varyantı](assets/screenshots/04-bes-varyant.png)

`v1_clahe` · `v2_shading` (aydınlatma düzeltme) · `v3_gradient` (kabartma kenarları) ·
`v4_morph` (morfolojik) · `v5_negative`. Tek varyant her alanda kazanmıyor — o yüzden
ensemble oylaması yapılıyor. Kırpma sonrası tek kesit:

![Kırpma + CLAHE](assets/screenshots/03-kirpma-clahe.png)

### 3. Döşeme (tiling) — kritik teknik bulgu

![Tiling](assets/screenshots/05-tiling.png)

Şerit (~20:1 en/boy) doğrudan motora verilince RapidOCR tespiti **tamamen atlıyor** (0 kutu) —
sebebi `width_height_ratio` eşiği (varsayılan 8). Çözüm: bindirmeli parçalara bölme
(`bant_yüksekliği × 6` güvenlik payıyla). Bu düzeltme okumayı sıfırdan normale döndürdü.

### 4. OCR kutuları + alan çıkarımı

![OCR kutuları](assets/screenshots/06-ocr-kutulari.png)

Örnek ham okuma: `155/70R13T` (0.88) · `RADIAL` (0.78) · `OUTSIDE-155/70R13·75T` (0.84) ·
`TUBELESS` (0.88). Regex + sözlük (rapidfuzz fuzzy) ile 9 alana dönüştürülüp
yanak × tile × varyant × motor kaynaklarından **ağırlıklı oylama** ile birleştiriliyor.

### 5. Varyant karşılaştırma (3 örnek lastik)

![Varyant karşılaştırma](assets/screenshots/07-varyant-karsilastirma.png)

---

## Sonuçlar (119 lastik, 30 lastiklik etiketli sette doğrulandı)

| Konfigürasyon | Süre (119 lastik) | sn/lastik | Doğruluk (30-GT, 8 alan ort.) |
|---|---|---|---|
| Faz 0: RapidOCR+EasyOCR, GPU (ilk ensemble) | 39.6 dk | 20.0 | %56.3 |
| **Faz 1: + sözlük/tutarlılık/DOT düzeltmeleri (seçilen)** | **37.7 dk** | **19.0** | **%63.2** |
| PP-OCRv6 tek motor (yüksek-doğruluk alternatifi) | 107.2 dk | 54.1 | %71.8 |

Alan bazında (seçilen konfigürasyon):

| Alan | Doğruluk (30-GT) | Kapsam (119 lastik) |
|---|---|---|
| Marka | %80 | %92 |
| Desen | %58 | %88 |
| Ebat | %72 | %81 |
| Hız grubu | %52 | %68 |
| Mevsim | %65 | %78 |
| DOT (üretim tarihi) | %73 | %77 |
| Üretildiği yer | %30 | %45 |
| Tüplü/Tüpsüz | %76 | %88 |

Motor karşılaştırması (9 lastik, sabit varyant): **RapidOCR** (272 sn) ≈ EasyOCR doğruluğunda ama
**3× hızlı**; Tesseract kabartma yazıda pratikte işe yaramıyor (%0–11). Detay: [`docs/RAPOR.md`](docs/RAPOR.md).

---

## Hızlı başlangıç

### Seçenek A — Docker (önerilen, kurulum derdi yok)

```bash
git clone https://github.com/yavuzzaltay/LastikOCR.git
cd LastikOCR

# Kendi lastik fotoğraflarını bu klasörlere koy:
#   Lastik_fotolari/Lastigin_ustu/Ust_1.jpg, Ust_2.jpg, ...
#   Lastik_fotolari/Lastigin_alti/Alt_1.jpg, Alt_2.jpg, ...

docker compose up --build
```

Tarayıcıda **http://localhost:5000** — fotoğraf seç, "Seçilenleri İşle"ye bas,
adım adım görseller + çıkarılan alanlar + süre bilgisini gör. Detay: [`docs/DOCKER.md`](docs/DOCKER.md)
(GPU ile çalıştırma dahil).

### Seçenek B — Yerel Python

```bash
pip install -r requirements.txt   # GPU yoksa: requirements-docker.txt önerilir

python src/webapp.py              # web arayüzü → http://127.0.0.1:5000
python src/benchmark.py           # motor × varyant karşılaştırması (etiketli set)
python src/run_all.py             # 119 lastiğin tamamını işle → results/sonuclar.csv
python src/run_all.py --engines=rapidocr_v6 --variants=v1_clahe  # PP-OCRv6 alternatifi
```

> Not: `Lastik_fotolari/` (~350 MB) `.gitignore`'dadır, repoda yoktur. Kendi görüntülerinle
> aynı dosya adlarıyla (`Ust_N.jpg` / `Alt_N.jpg`) çalıştırabilirsin.

---

## Proje yapısı

```
LastikOCR/
├── src/
│   ├── preprocess.py      # orientasyon, kalite kapısı, bant kırpma, 5 varyant
│   ├── tiling.py          # bindirmeli döşeme (width/height oranı tuzağının çözümü)
│   ├── engines/           # ortak arayüz: rapidocr, rapidocr_v6, easyocr, tesseract
│   │   └── base.py        # GPU otomatik algılama (gpu_available)
│   ├── extract.py         # regex + fuzzy sözlük ile alan çıkarımı
│   ├── lexicon.py         # marka/desen/mevsim sözlükleri
│   ├── fuse.py            # çok kaynaklı ağırlıklı oylama + marka↔desen tutarlılığı
│   ├── pipeline.py        # uçtan uca hat (benchmark + run_all + webapp paylaşır)
│   ├── scoring.py         # GT karşılaştırma
│   ├── benchmark.py       # motor × varyant karşılaştırması
│   ├── run_all.py         # tam veri seti koşumu (otomatik konfig seçimi)
│   ├── vlm.py             # yerel VLM denemesi (Qwen2.5-VL, fallback önerisi)
│   ├── webapp.py          # Flask arayüzü (foto seçimi + adım adım görselleştirme)
│   ├── templates/         # web arayüzü şablonları
│   ├── gen_pipeline_walkthrough.py  # README'deki adım görsellerini üretir
│   └── gen_debug_images.py          # örnek varyant görsellerini üretir
├── tests/
│   └── test_preprocess_manual.py    # hızlı görsel doğrulama
├── data/
│   └── ground_truth.csv   # 30 lastiklik etiketli set
├── results/
│   ├── sonuclar.csv       # seçilen konfigürasyonun 119 lastiklik çıktısı
│   ├── benchmark_results.csv / benchmark_log.json
│   ├── rapor.md / rapor.html       # tam fizibilite raporu
│   ├── debug/pipeline/    # adım adım görseller + walkthrough.json
│   └── archive/           # ara konfigürasyon çıktıları
├── docs/
│   ├── RAPOR.md           # fizibilite raporu (kopya)
│   └── DOCKER.md          # Docker ile çalıştırma (CPU + GPU)
├── assets/screenshots/    # README görselleri
├── Dockerfile / Dockerfile.gpu / docker-compose.yml / docker-compose.gpu.yml
└── requirements.txt / requirements-docker.txt / requirements-gpu.txt
```

---

## Yöntem (kısa)

1. **Orientasyon:** dikey/yatay + 180° belirsizliği, iki adayın OCR-güven skoruyla oylanması.
2. **Kalite kapısı:** satır parlaklık profilinde bant yoksa `UNREADABLE` (boş çekimleri eliyor).
3. **Bant kırpma + 5 kontrast varyantı** (CLAHE, aydınlatma düzeltme, gradyan, morfoloji, negatif).
4. **Bindirmeli tiling** ile OCR motorlarına uygun parça boyutu.
5. **3 OCR motoru** (RapidOCR, EasyOCR, Tesseract) + PP-OCRv6 alternatifi, ortak arayüzden.
6. **Alan çıkarımı:** regex (ebat, DOT, üretim yeri, tüp) + fuzzy sözlük (marka, desen) + 3 kaynaklı mevsim.
7. **Birleştirme:** güven + tekrar sayısıyla ağırlıklı oylama, marka↔desen tutarlılık cezası.

---

## Bilinen sınırlamalar & sonraki adımlar

- **DOT** en zor alan: oval damga tespiti (Hough-circle) + hedefli yüksek çözünürlüklü OCR öneriliyor.
- **Sözlük genişletme** en ucuz kazanç: envanterdeki tüm marka/desen listesi içe aktarılabilir.
- **VLM fallback:** Qwen2.5-VL doğru kesitte kusursuz okudu (DOT dahil) — klasik OCR'ın boş bıraktığı
  alanlara hedefli fallback olarak öneriliyor, tam pipeline olarak değil.
- Fiziksel tarafta **eğik (raking) aydınlatma** kontrastı en çok artıracak tek değişiklik.

Detaylı analiz: [`docs/RAPOR.md`](docs/RAPOR.md).

---

## Lisans

MIT — detay için `LICENSE` dosyasına bakın.
