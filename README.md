# 🍓 Strawberry Vision - Çilek Görüntü Analiz Sistemi

Google Colab uyumlu, katmanlı mimariye sahip profesyonel çilek tespit ve olgunluk sınıflandırma sistemi.

## 🎯 Özellikler

- ✅ YOLOv8 tabanlı çilek tespiti
- ✅ Olgunluk sınıflandırması (ripe, semi-ripe, unripe)
- ✅ Nesne takibi (tracking)
- ✅ Otomatik sayım ve istatistik
- ✅ Görselleştirme ve sonuç kaydetme
- ✅ Katmanlı mimari (Domain-Driven Design)
- ✅ Google Colab desteği
- ✅ Kapsamlı test coverage

## 🚀 Hızlı Başlangıç

### Kurulum

```bash
# Repository'yi klonla
git clone <repository-url>
cd SmartFarmBerry

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### Kullanım

```bash
# Tek görüntü ile çalıştır
python -m strawberry_vision.main --image sample.jpg --model path/to/best.pt

# Video ile çalıştır
python -m strawberry_vision.main --video video.mp4 --model path/to/best.pt --max-frames 100

# Smoke test
python tests/smoke_test.py
```

### Google Colab – Hızlı Başlangıç

Aşağıdaki adımlarla Google Colab üzerinde hızlıca eğitim ve inference çalıştırabilirsiniz.

#### Yöntem 1: Notebook ile Manuel Çalıştırma

- **1) Colab'i aç ve GPU seç**
  - Runtime > Change runtime type > Hardware accelerator: GPU

- **2) Depoyu Colab'e klonla**
  ```bash
  !git clone https://github.com/emrah1982/SmartFarmStrawberry.git
  %cd SmartFarmStrawberry
  ```

- **3) Bağımlılıkları kur**
  ```bash
  !pip install -q -r requirements.txt
  ```

- **4) Google Drive'ı bağla (checkpoint ve sonuçlar için)**
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

- **5) Roboflow API Key'i güvenli şekilde ayarla (ÖNEMLİ)**
  
  **Önerilen: Colab Secrets kullanın**
  ```python
  from google.colab import userdata
  import os
  
  # Sol panelde 🔑 (Secrets) ikonuna tıklayın
  # Name: ROBOFLOW_API_KEY, Value: rf_... (API key'iniz)
  os.environ['ROBOFLOW_API_KEY'] = userdata.get('ROBOFLOW_API_KEY')
  ```
  
  **Alternatif: Manuel giriş (geçici)**
  ```python
  from getpass import getpass
  import os
  
  API_KEY = getpass("Roboflow API Key: ")  # Girdiğiniz görünmez
  os.environ['ROBOFLOW_API_KEY'] = API_KEY
  ```
  
  🔑 API Key alma: https://app.roboflow.com/settings/api

- **6) Production notebook'u aç**
  - Dosya: `StrawberryVision_Colab_Production.ipynb`
  - İçerikte şunlar hazırdır:
    - Roboflow API ile dataset indirme (4 doğrulanmış dataset seçeneği)
    - Sınıf etiketlerini otomatik standardize etme
    - Eğitim konfigürasyonu (`configs/train_config.yaml`) ve augmentasyon ayarları
    - Her 10 epoch'ta checkpoint kaydetme (Google Drive)

- **7) Tüm hücreleri sırayla çalıştır**
  - Eğitim sonunda en iyi model ve tüm checkpoint'ler Drive'a kopyalanır.
  - Sonuç görselleri ve metrikler `runs/train/...` altında da kaydedilir.

#### Yöntem 2: Headless Çalıştırma (nbconvert)

Notebook'u dosya menüsünü açmadan komut satırından çalıştırabilirsiniz:

```python
# 1) Kurulum
!git clone https://github.com/emrah1982/SmartFarmStrawberry.git
%cd SmartFarmStrawberry
!pip install -q -r requirements.txt nbconvert jupyter roboflow

# 2) API Key'i ayarla (Colab Secrets'tan)
from google.colab import userdata, drive
import os

os.environ['ROBOFLOW_API_KEY'] = userdata.get('ROBOFLOW_API_KEY')
drive.mount('/content/drive')

# 3) Notebook'u çalıştır
!jupyter nbconvert --to notebook --execute StrawberryVision_Colab_Production.ipynb \
  --output executed.ipynb --ExecutePreprocessor.timeout=-1
```

#### Dataset Versiyonları

Roboflow datasetlerinin çoğu **version 2** veya üstünü kullanır. Eğer version hatası alırsanız:

```python
# Hücre 0'da VERSION parametresini değiştirin
VERSION = 2  # veya 3, 4, vb.
```

Mevcut versiyonları kontrol etmek için: `https://universe.roboflow.com/{workspace}/{project}`

**⚠️ Güvenlik Notu**: API key'inizi asla kod hücresine yazmayın. Colab Secrets veya `getpass()` kullanın.

Not: Colab dışında lokalde çalıştırmak için de aynı dizin yapısı ve `scripts/` altındaki yardımcı komutlar kullanılabilir.

## 📦 Model Eğitimi

### 1. Dataset Hazırlama

```bash
# Roboflow'dan dataset indir
python scripts/download_dataset.py --api-key YOUR_KEY --workspace strawberry --project ripeness

# Sınıf etiketlerini standardize et
python scripts/relabel_dataset.py --input datasets/roboflow --output datasets/processed

# Augmentation uygula (opsiyonel)
python scripts/augment_dataset.py --input datasets/processed --output datasets/augmented --factor 2
```

### 2. Model Eğitimi

```bash
# Config dosyası ile eğitim
python scripts/train_yolo.py --data configs/strawberry_data.yaml --config configs/train_config.yaml

# Komut satırı parametreleri ile
python scripts/train_yolo.py --data datasets/processed/data.yaml --epochs 100 --batch 16 --model yolov8s.pt
```

### 3. Model Değerlendirme

```bash
python scripts/evaluate_model.py --model runs/train/strawberry_exp/weights/best.pt --data configs/strawberry_data.yaml
```

## 🏗️ Proje Yapısı

```
SmartFarmBerry/
├── strawberry_vision/           # Ana uygulama paketi
│   ├── presentation/            # Görselleştirme katmanı
│   │   └── visualizer.py
│   ├── application/             # Uygulama katmanı
│   │   └── pipeline.py
│   ├── domain/                  # Domain katmanı
│   │   ├── entities.py
│   │   └── services.py
│   ├── infrastructure/          # Altyapı katmanı
│   │   ├── detectors.py
│   │   └── sources.py
│   └── main.py                  # Giriş noktası
│
├── configs/                     # Konfigürasyon dosyaları
│   ├── strawberry_data.yaml     # Dataset config
│   ├── train_config.yaml        # Eğitim parametreleri
│   └── augmentation_config.yaml # Augmentation ayarları
│
├── scripts/                     # Yardımcı scriptler
│   ├── download_dataset.py      # Dataset indirme
│   ├── relabel_dataset.py       # Etiket güncelleme
│   ├── augment_dataset.py       # Augmentation
│   ├── train_yolo.py            # Model eğitimi
│   └── evaluate_model.py        # Model değerlendirme
│
├── tests/                       # Test dosyaları
│   ├── test_domain_entities.py
│   ├── test_domain_services.py
│   ├── test_application_pipeline.py
│   └── smoke_test.py
│
├── docs/                        # Dokümantasyon
│   ├── INDEX.md                 # Dokümantasyon ana sayfa
│   ├── USAGE.md                 # Kullanım kılavuzu
│   ├── architecture.md          # Mimari tasarım
│   ├── development-rules.md     # Geliştirme kuralları
│   ├── 1-gorunuAnalizi.md       # Dataset stratejisi
│   ├── 2-YOLOegitimiHiperparametre.md
│   ├── 2.1-roboflowEtiketlemeTalimati.md
│   ├── 2.2-ModelHataAnaliziIyilestirmePromptu.md
│   └── 3-RoboflowDatasetKullanimi.md
│
├── requirements.txt             # Python bağımlılıkları
├── Colab_Starter.ipynb          # Colab notebook
└── README.md                    # Bu dosya
```

## 🧪 Test

```bash
# Tüm testleri çalıştır
pytest tests/

# Coverage ile
pytest --cov=strawberry_vision tests/

# Belirli bir test dosyası
pytest tests/test_domain_entities.py -v
```

## 📚 Dokümantasyon

Detaylı dokümantasyon için `docs/INDEX.md` dosyasına bakın:

- **Kullanım Kılavuzu**: Kurulum, çalıştırma, örnekler
- **Mimari Tasarım**: Katmanlı mimari, bağımlılıklar, veri akışı
- **Geliştirme Kuralları**: SOLID prensipleri, kod stili, test stratejisi
- **Model Eğitimi**: Dataset hazırlama, eğitim, değerlendirme
- **Roboflow Kullanımı**: Dataset linkleri, augmentation, best practices

## 🎨 Katmanlı Mimari

Proje Domain-Driven Design prensiplerine göre 4 katmana ayrılmıştır:

### 1. Domain Katmanı
- **entities.py**: `Ripeness`, `Detection`, `Strawberry` varlıkları
- **services.py**: `TrackingService`, `CountingService`
- Saf iş kuralları, harici bağımlılık yok

### 2. Infrastructure Katmanı
- **detectors.py**: YOLO detector, ripeness classifier
- **sources.py**: `ImageSource`, `VideoSource`, `CameraSource`
- Model, veri kaynakları, I/O işlemleri

### 3. Application Katmanı
- **pipeline.py**: `InferencePipeline`
- Orkestrasyon, loglama, metrik toplama
- Katmanlar arası koordinasyon

### 4. Presentation Katmanı
- **visualizer.py**: `Visualizer`
- Bounding box çizimi, sonuç kaydetme, overlay

## 🔧 Konfigürasyon

### Dataset Config (strawberry_data.yaml)
```yaml
path: ../datasets/strawberry_processed
train: images/train
val: images/val
nc: 3
names:
  0: strawberry_ripe
  1: strawberry_semi_ripe
  2: strawberry_unripe
```

### Eğitim Config (train_config.yaml)
```yaml
model: yolov8n.pt
epochs: 100
batch: 16
imgsz: 640
optimizer: AdamW
lr0: 0.01
# ... (detaylar için config dosyasına bakın)
```

## 📊 Sınıf Tanımları

- **strawberry_ripe**: Olgun çilek (kırmızı renk baskın, hasada hazır)
- **strawberry_semi_ripe**: Yarı olgun çilek (kırmızı-beyaz karışımı)
- **strawberry_unripe**: Olgun olmayan çilek (yeşil veya açık beyaz)

## 🌐 Roboflow Dataset Linkleri

Önerilen datasetler için `docs/3-RoboflowDatasetKullanimi.md` dosyasına bakın:
- Strawberry Detection Dataset
- Strawberry Ripeness Classification
- Fruit Detection - Strawberry
- Agricultural Strawberry Dataset

## 🤝 Katkıda Bulunma

1. Kod yazarken `docs/development-rules.md` kurallarına uyun
2. Her değişiklik için test yazın
3. Docstring ve type hint ekleyin
4. SOLID prensiplerine uyun
5. Katman sınırlarını ihlal etmeyin

## 📝 Lisans

[Lisans bilgisi eklenecek]

## 📧 İletişim

[İletişim bilgisi eklenecek]

## 🙏 Teşekkürler

- Ultralytics (YOLOv8)
- Roboflow (Dataset platformu)
- OpenCV
- Albumentations

---

**Not**: Detaylı kullanım ve geliştirme bilgileri için `docs/` klasöründeki dokümantasyonu inceleyin.
