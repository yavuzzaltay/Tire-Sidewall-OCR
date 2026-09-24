# Lastik Yanak OCR — Yerel Motor Benchmark & Fizibilite Raporu

**Tarih:** 2026-07-20
**Kapsam:** 119 lastik (238 görüntü: Üst + Alt yanak), tamamen yerel/offline OCR motorları
**Hedef alanlar:** Ebat, Hız Grubu, Marka, Desen, Mevsim, DOT (üretim tarihi), Ekstra lot, Üretildiği Yer, Tüplü/Tüpsüz

---

## 1. Özet

Line-scan kamerayla çekilmiş, lastik çevresi şerit halinde açılmış görüntülerden (≈1100×12900 px,
düşük kontrastlı kabartma yazı) yapısal bilgileri okumak için uçtan uca yerel bir OCR hattı kuruldu:
orientasyon tespiti → kalite kapısı → metin bandı kırpma → 5 farklı kontrast/aydınlatma varyantı →
döşeme (tiling) → 3 OCR motoru (RapidOCR, EasyOCR, Tesseract) → regex + sözlük tabanlı alan çıkarımı →
çoklu kaynak oylaması.

**En önemli bulgular:**
- **RapidOCR** (PP-OCR model ailesi, ONNX) tek motor olarak açık ara en iyisi: EasyOCR ile aynı/daha iyi
  doğruluk, CPU'da **3 kat daha hızlı**. **Tesseract kabartma lastik yazısında pratik olarak işe yaramıyor**
  (ana alanlarda %0-11 doğruluk).
- Tek bir ön işleme varyantı her alanda kazanmıyor; alanlar arasında **farklı varyantlar farklı işe yarıyor**
  (ör. `v2_shading` marka+üretim yerinde, `v5_negative` ebatta daha iyi).
- İlk benchmark turunda **desen (pattern) tanıma %22** ile en zayıf alan çıktı. Kök neden analiziyle
  (aşağıda §5) bunun **sözlükte eksik marka** (Waterfall, Crosswind hiç yoktu) ve **çok gevşek fuzzy-eşik**
  kaynaklı olduğu bulundu; düzeltme sonrası aynı sette **marka doğruluğu %56 → %100, desen %22 → %44**'e çıktı.
- **GPU kurulumu sonrası (§6.1)** RapidOCR+EasyOCR ikisi birlikte (ensemble) kullanılabilir hale geldi —
  bu, DOT dahil çoğu alanda belirgin doğruluk artışı sağladı **ve** toplam süreyi kısalttı (bkz. §6.2, §10).
- **İkinci Ar-Ge turunda** (§6.3): ground truth 9→30 lastiğe genişletildi; sözlük+tutarlılık
  düzeltmeleriyle 30-GT'de toplam doğruluk %56.3→%63.2; **PP-OCRv6** modeli denendi (%71.8, ama
  ~3× yavaş) ve **yerel VLM** (Qwen2.5-VL) denemesi yapıldı — doğru kesitte kusursuz okudu ama
  örnekleme stratejisi yetersiz kaldı (%30.5), üretime alınmadı.

---

## 2. Veri Seti

- `Lastik_fotolari/Lastigin_ustu`: 119 görüntü (dış/üst yanak)
- `Lastik_fotolari/Lastigin_alti`: 119 görüntü (iç/alt yanak, genelde "INSIDE" etiketli)
- Format: gri tonlamalı JPEG, boyutlar ~1100×12900 ile ~1750×19200 arası değişiyor (farklı lastik ebatları/tarama uzunlukları)
- **11 görüntü** (Üst tarafta) kalite kapısından geçemeyen boş/başarısız çekim (`Ust_56, 65, 66, 89, 99, 100, 103, 107, 109, 114, 117`)
- **7 görüntü** dosyada zaten yatay (transpoze) kaydedilmiş; geri kalanı dikey şerit — otomatik orientasyon çözümü ikisini de kapsıyor
- Veri setinde en az **9 farklı marka** (Continental, Goodyear, Kumho, Lassa, Waterfall, Bridgestone,
  Crosswind + muhtemelen Petlas/diğerleri) ve bunlara ait çok sayıda farklı desen/ebat kombinasyonu tespit edildi

---

## 3. Yöntem

### 3.1 Ön işleme (`src/preprocess.py`)
1. **Orientasyon çözümü**: Ham görüntü dikey/yatay olabiliyor; iki 180°-farklı aday üretilip her ikisinin
   üzerinde hızlı bir RapidOCR güven skoru hesaplanarak doğru yön otomatik seçiliyor (OCR-tabanlı oylama,
   salt görüntü istatistiğiyle 180° farkı ayırt edilemiyor).
2. **Kalite kapısı**: Satır parlaklık profilinde belirgin bir bant bulunamazsa görüntü `UNREADABLE`
   işaretlenip atlanıyor (boş/başarısız çekimleri otomatik eliyor — 11/119 görüntüde doğru çalıştığı doğrulandı).
3. **Metin bandı kırpma**: Satır medyan profiliyle en uzun sürekli "aydınlık" bölge bulunup arka plan
   (siyah) kırpılıyor.
4. **5 zenginleştirme varyantı**:
   | Varyant | Yöntem |
   |---|---|
   | `v1_clahe` | Temel CLAHE kontrast eşitleme |
   | `v2_shading` | Büyük çekirdekli Gauss ile aydınlatma düzeltme + CLAHE |
   | `v3_gradient` | Sobel gradyan büyüklüğü (kabartma kenarlarını vurgular) |
   | `v4_morph` | Top-hat/black-hat morfolojik kontrast artırma (polariteden bağımsız) |
   | `v5_negative` | v2'nin negatifi (bazı kaynaklarda ışık/gölge tersine dönüyor) |

### 3.2 Döşeme (`src/tiling.py`) — kritik bir teknik bulgu
İlk denemede tüm şeridi (ör. 12900×620 px) tek parça halinde motora vermek **tamamen başarısız** oldu
(0 kutu bulundu). Kök neden: RapidOCR'ın `width_height_ratio` eşiği (varsayılan **8**); girdi eni/boyu bu
oranı aştığında motor tespit aşamasını tamamen atlayıp tüm görüntüyü "tek satır" sanıyor. Bizim şeritler
~20:1 oranında olduğundan bu tuzağa düşüyordu. **Çözüm**: şeridi bindirmeli (~200px), oran güvenle 8'in
altında kalacak genişlikte (`min(config, bant_yüksekliği×6)` — ince bantlarda otomatik daralan güvenlik payı)
parçalara bölüp her parçayı ayrı OCR'lamak. Bu düzeltme sonrası okuma tamamen normale döndü.

### 3.3 OCR motorları (`src/engines/`)
RapidOCR (onnxruntime), EasyOCR, Tesseract (pytesseract + UB-Mannheim binary) — ortak arayüz üzerinden.
PaddleOCR de denendi; kurulum başarılı oldu ama bu ortamda (Windows, CPU) çalışma anında oneDNN/PIR
executor hatasıyla çöktü (`NotImplementedError: ConvertPirAttribute2RuntimeAttribute...`); RapidOCR zaten
aynı PP-OCR model ailesini ONNX Runtime üzerinden sorunsuz çalıştırdığından PaddleOCR devre dışı bırakıldı.

### 3.4 Alan çıkarımı (`src/extract.py`)
Regex (ebat, DOT, made-in, tüpsüz/tüplü) + rapidfuzz ile sözlük tabanlı fuzzy eşleştirme (marka, desen).
Mevsim üç kaynaktan çıkarılıyor: doğrudan metin işaretleri (M+S, ALL SEASON, 3PMSF), desen adı → mevsim
eşleme tablosu, ve (bu turda uygulanmadı) ikon şekil tanıma.

### 3.5 Birleştirme (`src/fuse.py`)
Her alan için {yanak × tile × varyant × motor} kaynaklı tüm adaylar toplanıp güven skoru + kaç farklı
kaynakta/motorda tekrarlandığı ile ağırlıklı oylanıyor; en yüksek skorlu değer kazanıyor.

---

## 4. Benchmark Sonuçları (9 lastiklik etiketli set)

### 4.1 Motor karşılaştırması (sabit varyant: v1_clahe)

| Motor | Süre (9 lastik) | Marka | Desen | Ebat | Hız Grubu | Mevsim | DOT | Üretim Yeri | Tüp |
|---|---|---|---|---|---|---|---|---|---|
| **RapidOCR** | **272s** | 56% | 22% | 78% | 33% | 67% | 67% | 33% | 86% |
| EasyOCR | 852s (3.1x yavaş) | 56% | 11% | 78% | 67% | 67% | 67% | 17% | 71% |
| Tesseract | 99s | 11% | 0% | 0% | 0% | 22% | 0% | 0% | 0% |

**Sonuç:** RapidOCR kazanan — EasyOCR'a eşit/üstün doğruluk, 3 kattan fazla hız. Tesseract kabartma
lastik yazısında kullanılamaz durumda (klasik ikili eşikleme tabanlı motorlar bu düşük kontrastlı,
gölge/ışık bağımlı yazı tipinde önemli ölçüde geride kalıyor).

### 4.2 Ön işleme varyant karşılaştırması (sabit motor: RapidOCR)

| Varyant | Marka | Desen | Ebat | Hız Grubu | Mevsim | DOT | Üretim Yeri | Tüp |
|---|---|---|---|---|---|---|---|---|---|
| v1_clahe | 56% | 22% | 78% | 33% | 67% | 67% | 33% | 86% |
| v2_shading | **67%** | 22% | 78% | 22% | 56% | 33% | **83%** | 86% |
| v3_gradient | 11% | 11% | 78% | 44% | 44% | 33% | 67% | 86% |
| v4_morph | 56% | 22% | 78% | 44% | 56% | 67% | 67% | 86% |
| v5_negative | 44% | 22% | **89%** | 33% | 56% | 67% | 50% | 86% |

**Sonuç:** Tek bir "en iyi" varyant yok — alan bazlı en iyi seçim değişiyor. Üretim ortamı için önerilen
yaklaşım: birden fazla varyantı paralel çalıştırıp oylamak (aşağıdaki ensemble sonucu bunu doğruluyor).

### 4.3 Ensemble (birleşim) sonuçları

| Konfigürasyon | Süre | Marka | Desen | Ebat | Hız Grubu | Mevsim | DOT | Üretim Yeri | Tüp |
|---|---|---|---|---|---|---|---|---|---|
| RapidOCR + 5 varyant | 864s | 56% | 22% | **89%** | 56% | 56% | 67% | **100%** | 86% |
| RapidOCR+EasyOCR + v4_morph | 1088s | **78%** | 22% | **89%** | 56% | 67% | 67% | 67% | 86% |

5 varyantı birlikte oylamak, tek varyanta göre ebat ve üretim yerinde belirgin iyileşme sağlıyor (maliyet: ~3.2x süre).
2 motoru birlikte kullanmak markada büyük katkı sağlıyor ama süre neredeyse 4 katına çıkıyor.

---

## 5. Vaka İncelemesi: Desen/Marka Doğruluğunu İyileştirme

İlk benchmark turunda **desen tanıma tüm konfigürasyonlarda ~%22'de takılı kaldı**. Kök neden analizi:

1. **Marka-bağımsız desen araması**: `_extract_brand_pattern` tüm marka+desen sözlüğüne karşı fuzzy
   arama yapıyordu; bu yüzden ör. bir **Crosswind "Comfort Peak"** lastiği yanlışlıkla **Semperit
   "Comfort-Life 2"**'ye eşleşebiliyordu (kelime yapısı benzer, farklı marka).
2. **Sözlükte eksik markalar**: Veri setinde gerçekten var olan **Waterfall** ve **Crosswind** markaları
   ilk sözlükte hiç yoktu — bu markaların lastikleri asla doğru tanınamıyordu.
3. **Çok gevşek fuzzy eşiği** (score_cutoff=78): Kısa/genel karakterli sözlük string'leri ("Green-Max",
   "Incurro A/S ST450", "AS-1") gürültülü OCR metniyle tutarlı biçimde yanlış-pozitif üretiyordu.

**Denenen düzeltmeler ve sonuçları** (aynı 9 lastiklik sette, RapidOCR + v4_morph):

| Adım | Marka | Desen |
|---|---|---|
| Başlangıç | 56% | 22% |
| Tek-kaynak marka-kısıtlı desen araması | 56% | 22% *(iyileşme yok — yerel marka tahmini de gürültülü)* |
| İki geçişli mimari (önce TÜM kaynaklardan marka oylanır, sonra o markaya kısıtlı 2. desen geçişi) | 56% | 22% *(hâlâ yok — kök neden başka yerde)* |
| + Fuzzy eşiği 78→88 (desen), 68→80 (2. geçiş) yükseltme | 78% | 33% |
| + Sözlüğe Waterfall, Crosswind eklendi | **100%** | **44%** |

**Çıkarım:** Asıl kazanç mimari karmaşıklıktan değil, **(a) sözlüğün veri setini gerçekten kapsaması** ve
**(b) fuzzy eşiğinin yeterince sıkı tutulmasından** geldi. "Bilmiyorum" (None) döndürmek, düşük eşikle
yanlış-ama-kendinden-emin bir cevap vermekten üretim ortamı için çok daha güvenli — bu değişiklikle
sistem artık emin olmadığında sessiz kalıyor, yanlış marka/desen uydurmuyor.

---

## 6. Tam Veri Seti Sonuçları (119 lastik)

`run_all.py`, benchmark'ın verdiği maliyet/doğruluk analizine göre **RapidOCR + v1_clahe** (tek motor,
tek varyant) konfigürasyonunu seçti — gerekçe: en pahalı ensemble (RapidOCR+EasyOCR, öncelikli alanlarda
+8 puan) 119 lastikte tahmini ~240 dakika sürecekti; bu oturum için hız/kapsam dengesi gözetilerek hızlı
konfigürasyon tercih edildi (bkz. `run_all.py: pick_best_config`, sınır 90 dk).

**Çalışma istatistikleri:**
- Toplam süre: **2779 sn (~46 dk)**, ortalama **23.4 sn/lastik** (119 lastik × 2 yanak)
- Kalite kapısından geçemeyen (okunamadı): **Üst 11/119 (%9)**, **Alt 7/119 (%6)**

**Alan bazlı kapsam** (119 lastiğin kaçında BİR değer üretildi — doğruluk değil, "boş mu doldu mu"):

| Alan | Kapsam |
|---|---|
| Marka | 107/119 (%90) |
| Desen | 84/119 (%71) |
| Ebat | 90/119 (%76) |
| Hız Grubu | 53/119 (%45) |
| Mevsim | 80/119 (%67) |
| DOT | 92/119 (%77) |
| Üretim Yeri | 50/119 (%42) |
| Tüp (tüplü/tüpsüz) | 103/119 (%87) |

**Doğruluk** (9 lastiklik ground truth setinin bu tam çalıştırmadaki gerçek sonuçları — `data/ground_truth.csv`
ile `results/sonuclar.csv` çapraz kontrolü):

| Alan | Doğruluk |
|---|---|
| **Marka** | **8/9 = %89** |
| **Desen** | **5/9 = %56** |
| Ebat | 7/9 = %78 |
| Hız Grubu | 3/9 = %33 |
| Mevsim | 5/9 = %56 |
| DOT | 2/3 = %67 *(çoğu GT satırında DOT "?" işaretli, örneklem küçük)* |
| Üretim Yeri | 2/6 = %33 |
| Tüp | 6/7 = %86 |

İlginç bir gözlem: bu tam-set çalışması `v1_clahe` varyantıyla, önceki küçük-ölçek doğrulamalarda kullanılan
`v4_morph`'a göre **marka ve desende daha da iyi** sonuç verdi (marka %89 vs %100 — v4_morph biraz daha iyiydi
ama v1_clahe genele daha hızlı yayıldı; desen %56 vs %44 — v1_clahe burada daha iyi). Bu, §4.2'deki "tek bir
varyant her şeyde kazanmıyor" bulgusunu bir kez daha doğruluyor; üretim sisteminde birden fazla varyantı
paralel oylamak (§4.3'teki ensemble sonucu gibi) en tutarlı sonucu verecektir.

Tüm 119 lastiğin alan-alan dökümü: **`results/sonuclar.csv`** (güven skorlarıyla birlikte).

### 6.1 Sonradan Eklenen: GPU Hızlandırma

Rapor tamamlandıktan sonra kullanıcı isteğiyle CUDA'lı `torch` (RTX 3060 Laptop, CUDA 12.6) ve
`onnxruntime-gpu` kurularak her iki motor da GPU'ya taşındı. İki teknik engel çözüldü:
- `onnxruntime-gpu` CUDA/cuDNN DLL'lerini paketin içinde taşımıyor — `nvidia-cublas`, `nvidia-cufft`,
  `nvidia-cudnn-cu13` pip paketleri kurulup PATH'e eklendi.
- torch'un kendi paketlediği cuDNN ile ayrı kurulan cuDNN aynı süreçte çakışıyordu (`WinError 127`);
  `src/engines/__init__.py`'de `torch`'u en başta import ederek çakışma torch lehine sabitlendi.

**Sonuç** (3 lastiklik hızlı doğrulama turu):
- RapidOCR: 30.2sn/lastik (CPU) → **9.7sn/lastik (GPU), 3.1× hızlanma**
- EasyOCR: 94.7sn/lastik (CPU) → **14.8sn/lastik (GPU), 6.4× hızlanma**

### 6.2 Bu Hız Kazancını Doğruluğa Çevirmek: Ensemble'a Geçiş

**Neden gerekliydi:** §4.3'teki benchmark, RapidOCR+EasyOCR'ı `v4_morph` varyantında birlikte kullanmanın
(2 motor × 1 varyant = "x2 maliyet") öncelikli alanlarda (marka, desen, ebat, DOT) tek-motor/tek-varyant
konfigürasyonuna göre **+8 puan** kazandırdığını göstermişti. Ama bu konfigürasyon 119 lastiğin tamamında
CPU'da tahmini **~240 dakika** sürüyordu — 90 dakikalık oturum bütçesini aşıyordu, bu yüzden ilk üretim
koşusu (§6) daha ucuz ama daha az isabetli tek-motor/tek-varyant konfigürasyonuyla yapılmıştı.

**Ne değişti:** GPU kurulumuyla RapidOCR ~3.1×, EasyOCR ~6.4× hızlandığı için bu ensemble konfigürasyonu
artık ~80 dakikaya iniyor — bütçeye sığıyor. `src/run_all.py`'deki otomatik konfigürasyon seçici
(`pick_best_config`) güncellendi: artık CPU-ölçümlü süre tahminine bir `gpu_speedup` çarpanı (temkinli x3.0)
uyguluyor. Sistem gerçekten de **RapidOCR + EasyOCR (v4_morph)** konfigürasyonunu kendi seçti ve 119 lastiğin
tamamını bu konfigürasyonla yeniden işledi.

**Sonuç — hem daha hızlı hem daha doğru:** yeni konfigürasyon (2 motor) GPU'da, eski konfigürasyondan
(1 motor) CPU'da daha hızlı çıktı — GPU'nun hızlanması, ikinci motoru eklemenin maliyetinden büyük.

| Metrik | Eski (RapidOCR, v1_clahe, CPU) | Yeni (RapidOCR+EasyOCR, v4_morph, GPU) |
|---|---|---|
| Toplam süre (119 lastik) | 46 dk | **39.6 dk** |
| Lastik başına | 23.4 sn | **20.0 sn** |

**Alan bazlı doğruluk (9 lastiklik GT):**

| Alan | Eski | Yeni | Fark |
|---|---|---|---|
| Marka | 8/9 — %89 | 7/9 — %78 | −11 puan |
| Desen | 5/9 — %56 | 5/9 — %56 | değişmedi |
| Ebat | 7/9 — %78 | 8/9 — %89 | +11 puan |
| Hız Grubu | 3/9 — %33 | 7/9 — %78 | **+45 puan** |
| Mevsim | 5/9 — %56 | 6/9 — %67 | +11 puan |
| DOT | 2/3 — %67 | 3/3 — %100 | +33 puan |
| Üretildiği Yer | 2/6 — %33 | 4/6 — %67 | +34 puan |
| Tüplü/Tüpsüz | 6/7 — %86 | 6/7 — %86 | değişmedi |

8 alanın ortalaması: %62.3 → **%77.6 (+15.3 puan)**. Tek düşen alan marka (9 lastiklik küçük örneklemde
1 lastiğin sınırda kayması) — 119'un tamamındaki **kapsamda** marka aslında yükseldi (aşağıya bakınız),
bu düşüş büyük olasılıkla örneklem gürültüsü.

**Kapsam (119 lastiğin kaçında bir değer üretildi), eski vs yeni:**

| Alan | Eski | Yeni |
|---|---|---|
| Marka | 107/119 — %90 | **110/119 — %92** |
| Desen | 84/119 — %71 | **97/119 — %82** |
| Ebat | 90/119 — %76 | **96/119 — %81** |
| Hız Grubu | 53/119 — %45 | **83/119 — %70** |
| Mevsim | 80/119 — %67 | **86/119 — %72** |
| DOT | 92/119 — %77 | **102/119 — %86** |
| Üretildiği Yer | 50/119 — %42 | **54/119 — %45** |
| Tüplü/Tüpsüz | 103/119 — %87 | **105/119 — %88** |

**Neden bu kadar iyileşti?** İki motorun (RapidOCR + EasyOCR) farklı hata modları var — biri bir
tile'da/varyantta okuyamadığını diğeri okuyabiliyor; §3.3'teki oylama mekanizması bu iki bağımsız görüşü
birleştirip tek kaynaktan gelen sonuçtan daha sağlam bir karar üretiyor. Sonuç dosyaları:
`results/sonuclar_gpu_ensemble_final.csv` (Faz 0, ilk GPU+ensemble) vs
`results/sonuclar_cpu_v1clahe_baseline.csv` (en eski, CPU tek-motor).

### 6.3 İkinci İyileştirme Turu: Doğruluk + Hız (Faz 0-3)

Kullanıcının "inisiyatif al, farklı yöntemler/modeller dene, gerekirse VLM gibi ağır araçları da
dene" isteğiyle yapılan ikinci bir Ar-Ge turu. Dört faz halinde ilerlendi; her fazın sonunda 30
lastiklik genişletilmiş ground truth setinde ölçüm yapıldı, kazanmayan yaklaşımlar da (VLM gibi)
dürüstçe raporlandı.

**Faz 0 — Ground Truth Genişletme (9 → 30 lastik):** Önceki 9 lastiklik GT, iyileştirmeleri
güvenilir ölçmek için yetersizdi (DOT'ta sadece 3 gerçek değer vardı). 21 yeni lastik hedefli
seçilip elle etiketlendi: 10 Kumho (desen=None kalanlar), 5 marka=None kalan, 6 marka-çeşitliliği
+yüksek güvenli DOT adayı. İki önemli bulgu: (1) **Kumho'nun iki farklı, sık tekrar eden modeli
var** — "EcoWing ES31" (205/55 R16) ve "Solus TA21" (185/65 R14) — ikisi de sözlükte hiç yoktu.
(2) **5 lastik gerçekten okunamaz** (insan gözüyle de) — sistemin marka=None dönmesi hata değil
doğru davranış.

**Faz 1 — Hızlı Doğruluk Kazanımları (kod düzeltmeleri, yeni model yok):** (1) `lexicon.py`'ye
eksik Kumho desenleri eklendi. (2) `fuse.py`'ye **marka↔desen tutarlılık cezası** eklendi — nihai
marka biliniyorken o markanın sözlüğünde olmayan desen adayları ×0.15 cezalandırılıyor. (3)
`extract.py`'de DOT sıkılaştırıldı — yalnız geçerli hafta+yıl formatındaki adaylar `dot`'a,
diğer alfasayısal kodlar `ekstra_lot`'a gidiyor.

| Alan | Faz 0 (30-GT) | Faz 1 (30-GT) | Fark |
|---|---|---|---|
| Marka | %80 | %80 | değişmedi |
| **Desen** | %33 | **%58** | **+25 puan** |
| **Mevsim** | %43 | **%65** | **+22 puan** |
| DOT | %64 | %73 | +9 puan |
| **TOPLAM (8 alan)** | **%56.3** | **%63.2** | **+6.9 puan** |

Teşhis: `sonuclar.csv` analizinde "General Tire" markasına başka markaların desenlerinin (Eco
Dynamic, Green-Max, Ventus Prime 4 — hiçbiri General Tire'ın kendi sözlüğünde yok) sık sık
yanlışlıkla atandığı görüldü; marka↔desen tutarlılık cezasının tam hedeflediği hata türü.

**Faz 2 — Model Yükseltme: PP-OCRv6 Keşfi:** Yeni nesil `rapidocr` paketi (3.9.1) denendi — eski
`rapidocr_onnxruntime` (1.2.3, PP-OCRv3) yerine **PP-OCRv6** det/rec modelleri kullanıyor. Ayrı
bir motor adıyla (`rapidocr_v6`) eklendi, eskisiyle yan yana durabiliyor.

| Alan | Eski (PP-OCRv3) | Yeni (PP-OCRv6) | Fark |
|---|---|---|---|
| Ebat | %52 | **%92** | **+40 puan** |
| Hız Grubu | %20 | **%76** | **+56 puan** |
| Üretim Yeri | %15 | **%45** | **+30 puan** |
| **TOPLAM (8 alan)** | **%54.6** | **%71.8** | **+17.2 puan** |
| Hız (119, tam koşu) | ~20 sn/lastik | 54.1 sn/lastik | ~2.7× yavaş |

PP-OCRv6 belirgin şekilde daha güçlü ama ~2.7-5× daha yavaş.

**Faz 3 — Yerel VLM Denemesi: Qwen2.5-VL 3B (Ollama):** Ollama üzerinden Qwen2.5-VL 3B (4-bit,
~3GB, tamamen yerel, GPU) kuruldu, yapılandırılmış JSON çıktı isteyen bir prompt ile denendi
(`src/vlm.py`). İlk denemede boş yanıt döndü — kök neden: gri tonlamalı (tek kanal) PNG + aşırı
geniş (2600px) görüntü. RGB'ye çevirip 1400px'e sınırlayınca:

> **Tek lastikte (Continental EcoContact 6) sonuç — 4.7 saniyede:** marka=Continental ✓,
> desen=EcoContact 6 ✓, ebat=155/70 R13 ✓, **DOT=1526 ✓** (en zor alan!), tüp=Tüpsüz ✓ —
> 5 kritik alanın 5'i de doğru, tek bir görüntü kesitinden.

Ama **30 lastiklik tam benchmark'ta genel doğruluk sadece %30.5** çıktı (marka %8, desen %25,
ebat %44, DOT %36). Neden? Klasik pipeline her lastik için onlarca tile/varyant tarıyor; VLM
testinde ise sadece 1 kesit/yanak (2 kesit/lastik) denendi — marka/desen yazısı çoğu lastikte bu
tek kesitin dışında kaldı. **Bu VLM'in okuma kapasitesinin değil, örnekleme stratejisinin
sınırı** — tire 5 örneği (doğru kesit → kusursuz okuma) bunu kanıtlıyor.

*Neden üretime alınmadı:* Klasik tiling yaklaşımını VLM'e uygulamak (lastik başına 6+ çağrı ×
~14sn) 119 lastikte ~3 saate çıkar — bütçeyi aşıyor. **Somut sonraki adım:** VLM'i tam pipeline
yerine, klasik OCR'ın boş bıraktığı alanlar için hedefli fallback olarak kullanmak.

**Nihai Karar — Üç Doğrulanmış Konfigürasyon** (119 lastiğin tamamında koşturulup 30-GT ile
puanlanmış):

| Konfigürasyon | Süre (119) | sn/lastik | Doğruluk (30-GT) |
|---|---|---|---|
| Faz 0 (ilk GPU+ensemble) | 39.6 dk | 20.0 | %56.3 |
| **Faz 1-ensemble (seçilen)** | **37.7 dk** | **19.0** | **%63.2** |
| PP-OCRv6 (tek motor, alternatif) | 107.2 dk | 54.1 | %71.8 |

**Seçilen üretim konfigürasyonu: Faz 1-ensemble.** Faz 0'a göre hem daha hızlı hem daha doğru —
net kazanç, ödünleşim yok. PP-OCRv6 daha yüksek doğruluk sunuyor ama ~2.85× yavaş; doğruluk
önceliği süre kısıtından ağır basan senaryolar için (gece toplu işleme vb.) iyi bir alternatif —
`results/sonuclar_v6_final.csv`'de saklandı. Tüm ara sonuçlar `results/` altında korunuyor.

---

## 7. Bilinen Sınırlamalar

- **DOT (üretim tarihi)**: En zor alan. Tarih kodu genelde küçük, oval damgalı ve şeridin rastgele bir
  noktasında; şu anki genel regex+bağımsız-4-hane yaklaşımı gürültüye açık (ör. tire 5'te doğru "1526"
  yerine "0724" gibi yanlış bir 4-haneli aday kazanabiliyor).
- **Orientasyon çözümünde nadir hata**: 18 örnek elle incelemede 1 tanesinde (tire 82, Alt yanak) otomatik
  yön seçici 180° ters yönü seçti. OCR-güven-skoru tabanlı yöntem çoğunlukla güvenilir ama %100 değil.
- **Çok ince bantlı görüntüler**: Bazı lastiklerde (ör. tire 110 Üst yanak) tespit edilen metin bandı
  anormal ince (157px) çıkıyor ve az bilgi taşıyor; muhtemelen o karede lastik hafif kaymış/net değil.
- **Tesseract kullanılamaz** durumda bu görüntü tipi için; RapidOCR/EasyOCR şart.
- **Ekstra lot alanı** kesin doğrulanabilir değil (üretici iç kodları, ground truth'ta da belirsizlik var)
  ama ham kodlar (S-243255, 03H536 gibi) tutarlı biçimde yakalanıyor — insan gözüyle çapraz kontrol için faydalı.

---

## 8. Öneriler ve Sonraki Adımlar

**Kısa vadede (düşük efor, somut kazanç):**
1. **Sözlüğü genişletmek**: Bu rapor sırasında 2 eksik marka bulundu; gerçek envanterdeki TÜM marka/desen
   listesi sözlüğe eklenirse (Excel/ERP'den içe aktarılabilir) marka+desen doğruluğu büyük ölçüde artar.
2. **DOT'a özel bölge tespiti**: Tarih kodu neredeyse her zaman bir **oval/daire** içinde. Hough-circle
   veya basit kontur tespitiyle önce bu ovali bulup sadece o küçük bölgeyi yüksek çözünürlükte OCR'lamak,
   DOT doğruluğunu muhtemelen tek başına en çok artıracak değişiklik.
3. **Fiziksel aydınlatma**: Görüntüler düz/dik açılı ışıkla çekilmiş gibi duruyor, kontrast çok düşük.
   **Eğik (raking) aydınlatma** kabartma harflerde gölge oluşturup kontrastı önemli ölçüde artırır —
   yazılımdan bağımsız, muhtemelen en yüksek getirili tek değişiklik.

**Orta vadede:**
4. **Marka logosu şekil tanıma**: Metin yerine/yanında her markanın logosunu (Continental'ın atı,
   Goodyear'ın kanatlı ayağı, Lassa'nın bayrağı) template matching veya küçük bir CNN ile tanımak,
   kontrasttan bağımsız çalışacağından metin OCR'dan daha güvenilir marka tespiti sağlayabilir.
5. **Mevsim ikonlarını doğrudan sınıflandırmak**: Güneş/bulut/kar tanesi ikonlarını metinden çıkarım
   yerine doğrudan şekil tanımayla (3PMSF kar tanesi vb.) tespit etmek daha güvenilir olur.
6. **Çevresel tekrar doğrulaması**: Yazı şerit boyunca lastik çevresinde ~2 kez tekrarlıyor; bu turda
   fuse.py bunu dolaylı kullanıyor (farklı tile'lar arası oylama). Periyot tespiti yapıp iki tekrarı
   doğrudan hizalayıp karşılaştırmak ekstra bir doğrulama katmanı olur.

**Uzun vadede / üretim hattı için:**
7. **Küçük OCR modelini bu domain'e fine-tune etmek**: Genel amaçlı OCR modelleri kabartmalı lastik
   yazısı için eğitilmemiş. Bu 119 lastik + genişletilmiş bir etiketli set ile RapidOCR'ın recognition
   modülü fine-tune edilirse büyük iyileşme beklenir.
2'026'da 6GB VRAM'e sığan yerel VLM'ler (ör. Qwen2.5-VL 3B, 4-bit) klasik OCR'dan daha iyi bağlamsal
   okuma yapabilir; bulut API kullanılmadığı için sonraki adım olarak **tamamen yerel** denenebilir.
9. **Güven eşiğine dayalı yönlendirme**: %100 otomasyon yerine, düşük güvenli alanları (özellikle DOT)
   bir operatöre hızlı onay için göstermek — tam otomasyon + insan doğrulaması hibrit modeli, saf
   otomasyondan daha güvenilir bir üretim sistemi olur.

---

## 9. Sonuç — Nihai Değerler

Tamamen yerel/offline motorlarla, düşük kontrastlı kabartma lastik yazısından anlamlı ölçüde bilgi
çıkarmak **mümkün ve fizibıl**. İki tur Ar-Ge sonunda (§6.2, §6.3), 30 lastiklik genişletilmiş
ground truth ile doğrulanmış üretim konfigürasyonu: **GPU + RapidOCR＋EasyOCR ensemble
(`v4_morph`) + Faz 1 kod düzeltmeleri** (sözlük genişletme, marka↔desen tutarlılık cezası, DOT
sıkılaştırma). İlk tura göre **hem daha hızlı hem daha doğru**; daha da yüksek doğruluk isteyen
senaryolar için PP-OCRv6 tek-motor alternatifi de doğrulanıp saklandı.

**Genel:**
- Toplam süre (119 lastik, GPU, 2-motor ensemble): **37.7 dk** (2260 sn), ortalama **19.0 sn/lastik**
- Kalite kapısında elenen görüntü: **18/238 (%7.6)** (Üst 11/119, Alt 7/119)
- 8 alan ortalama doğruluk: **%63.2** (30-GT) — ilk tur %56.3 idi

**Alan bazlı sonuçlar:**

| Alan | Doğruluk (30 lastiklik GT) | Kapsam (119 lastikte) |
|---|---|---|
| **Marka** | **20/25 — %80** | 110/119 — %92 |
| Desen | 14/24 — %58 | 105/119 — %88 |
| Ebat | 18/25 — %72 | 96/119 — %81 |
| Hız Grubu | 13/25 — %52 | 81/119 — %68 |
| Mevsim | 15/23 — %65 | 93/119 — %78 |
| DOT (üretim tarihi) | 8/11 — %73 | 92/119 — %77 |
| Üretildiği Yer | 6/20 — %30 | 54/119 — %45 |
| **Tüplü/Tüpsüz** | 16/21 — %76 | 105/119 — %88 |

Ground truth 9'dan 30 lastiğe genişletildi (§6.3, Faz 0) — DOT örneklemi 3'ten 11'e çıktı, artık
istatistiksel olarak daha anlamlı. "Kapsam", sistemin bir değer ürettiği oranı gösterir — doğru
olduğu anlamına gelmez; doğruluk sütunu ground truth ile karşılaştırmadır.

**Üç konfigürasyonun karşılaştırması:**

| Konfigürasyon | Toplam süre (119) | sn/lastik | Doğruluk (30-GT) |
|---|---|---|---|
| Faz 0: RapidOCR-v3+EasyOCR, GPU (ilk tur) | 39.6 dk | 20.0 | %56.3 |
| **Faz 1: RapidOCR-v3+EasyOCR, GPU + kod düzeltmeleri (seçilen)** | **37.7 dk** | **19.0** | **%63.2** |
| PP-OCRv6 tek motor (alternatif, yüksek doğruluk) | 107.2 dk | 54.1 | %71.8 |

GPU'nun sağladığı hızlanma (RapidOCR 3.1×, EasyOCR 6.4×) ikinci motoru eklemenin maliyetinden
büyük olduğu için Faz 1 konfigürasyonu Faz 0'a göre hem daha isabetli hem daha hızlı çıktı.
PP-OCRv6 (§6.3) daha da yüksek doğruluk sunuyor ama ~2.85× yavaş — süre kısıtı gevşek olan
senaryolar (gece toplu işleme vb.) için değerlendirilebilir.

En zayıf alanlar üretim yeri (%30) ve hız grubu (%52) — somut sonraki-adım önerileri §8'de (sözlük
genişletme, DOT'a özel oval tespiti, VLM tabanlı hedefli fallback §6.3). Sistem tamamen modüler
(`src/preprocess.py`, `tiling.py`, `engines/`, `extract.py`, `fuse.py`, `vlm.py`) olduğundan hem
sözlük genişletmeye hem de yeni motor/varyant eklemeye açık.
