# 🌾 FarmVision - Akıllı Tarım Analiz Platformu

**YOLOv7 tabanlı AI ile meyve tespiti, NDVI analizi ve tarım haritalaması web uygulaması**

FarmVision, tarım sektörüne yönelik geliştirilmiş kapsamlı bir yapay zeka platformudur. YOLOv7 derin öğrenme modeli ile meyve tespiti, çoklu vejetasyon indeksleri ile bitki sağlığı analizi ve drone görüntüleri ile tarım haritalaması yapabilir.

## ✨ Ana Özellikler

### 🍎 Meyve Tespiti ve Sayma
- **🍊 Mandalina** - 0.125 kg/adet
- **🍎 Elma** - 0.105 kg/adet  
- **🍐 Armut** - 0.220 kg/adet
- **🍑 Şeftali** - 0.185 kg/adet
- **🍇 Nar** - 0.300 kg/adet
- **🌴 Hurma** - 0.050 kg/adet

### 📊 Vejetasyon İndeksleri ve Bitki Sağlığı Analizi
- **NDVI** - Normalized Difference Vegetation Index
- **GLI** - Green Leaf Index
- **VARI** - Visual Atmospheric Resistance Index
- **NDYI** - Normalized Difference Yellowness Index
- **NDRE** - Normalized Difference Red Edge Index
- **NDWI** - Normalized Difference Water Index
- **EVI** - Enhanced Vegetation Index
- **SAVI** - Soil Adjusted Vegetation Index
- **LAI** - Leaf Area Index
- **20+ ek vejetasyon indeksi** desteği

### 🗺️ Tarım Haritalaması ve Coğrafi Analiz
- **GeoTIFF işleme** - Coğrafi referanslı görüntü analizi
- **Leaflet.js Haritalar** - İnteraktif web haritaları
- **Multi-Layer Desteği** - OSM, Google Satellite, Hybrid görünümler
- **Georaster Görselleştirme** - Drone görüntülerinin harita üzerinde gösterimi
- **Raster analizi** - Çok bantlı görüntü işleme
- **RGB ve NIR** - Görünür ışık ve yakın kızılötesi analiz
- **Renk haritası uygulaması** - 15+ farklı renk paleti
- **Histogram analizi** - İstatistiksel görüntü analizi

### 🎨 Web Arayüzü Özellikleri
- **Farm Vision Brand** - Özel marka kimliği
- **Responsive Design** - Bootstrap tabanlı mobil uyumlu tasarım
- **Dashboard** - Gerçek zamanlı istatistik kartları
- **İnteraktif Menü** - Collapsible sidebar navigation
- **Multi-Language** - Türkçe arayüz desteği
- **Color Themes** - 10+ farklı renk teması
- **Profile Management** - Kullanıcı profil yönetimi
- **Drag & Drop** - Dosya yükleme desteği
- **Notification System** - Gerçek zamanlı bildirimler

## 🛠️ Teknoloji Stack'i

- **Backend**: Django 4.1
- **AI Model**: YOLOv7 (PyTorch 2.0.1)
- **Computer Vision**: OpenCV, PIL
- **Geospatial**: GDAL 3.4.1, Rasterio, Rio-tiler
- **Data Processing**: NumPy, Pandas, Openpyxl, Scipy
- **Drone Integration**: PyODM (OpenDroneMap)
- **Frontend**: HTML5, Bootstrap, JavaScript, Leaflet.js
- **Charts**: Chart.js, FullCalendar
- **Icons**: FontAwesome Free 6.2.1
- **Maps**: OpenStreetMap, Google Maps API
- **UI Libraries**: Sweet Alert, Waves Effect, mCustomScrollbar
- **Database**: SQLite / PostgreSQL
- **Deployment**: Docker, Gunicorn, Docker Compose
- **Utilities**: NatSort, Zipfile, Subprocess, NLTK

## 🚀 Kurulum

### Gereksinimler
- Python 3.8+
- CUDA (GPU kullanımı için, opsiyonel)
- Docker & Docker Compose (önerilen)
- GDAL (coğrafi veri işleme için)
- YOLOv7 model dosyaları (.pt)

### 🐳 Docker ile Kurulum (Önerilen)

1. **Repository'i klonlayın**
```bash
git clone https://github.com/skyfarmx/farmvision.git
cd farmvision
```

2. **Model dosyalarını yerleştirin**
Aşağıdaki YOLOv7 model dosyalarını proje ana dizinine yerleştirin:
- `mandalina.pt`
- `elma.pt`
- `armut.pt`
- `seftali.pt`
- `nar.pt`
- `hurma.pt`

3. **Docker Compose ile başlatın**
```bash
docker-compose up --build
```

4. **Uygulamaya erişin**
- Web arayüzü: http://localhost:8000
- OpenDroneMap: http://localhost:3000

### 📦 Manuel Kurulum

1. **Sistem bağımlılıklarını yükleyin (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install python3-pip python3-dev libpq-dev ffmpeg libsm6 libxext6 gdal-bin libgdal-dev
```

2. **Repository'i klonlayın**
```bash
git clone https://github.com/skyfarmx/farmvision.git
cd farmvision
```

3. **Virtual environment oluşturun**
```bash
pip install virtualenv
virtualenv myprojectenv --python=python3
source myprojectenv/bin/activate
```

4. **Gerekli paketleri yükleyin**
```bash
pip install -r requirements.txt
pip install 'numpy<2' torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

5. **Django migration'larını çalıştırın**
```bash
python manage.py migrate
```

6. **Superuser oluşturun**
```bash
python manage.py createsuperuser
```

7. **Sunucuyu başlatın**
```bash
python manage.py runserver
```

## 📱 Kullanım

### Tek Görüntü Analizi
1. Ana sayfaya gidin (`/`)
2. Meyve türünü seçin
3. Ağaç sayısını girin
4. Ağaç yaşını girin
5. Ekiliş sırasını belirtin
6. Görüntüyü yükleyin
7. "Analiz Et" butonuna tıklayın

### Çoklu Görüntü Analizi
1. Çoklu tespit sayfasına gidin (`/mcti`)
2. Meyve türünü seçin
3. Ekiliş sırasını belirtin (örn: 3-5)
4. Birden fazla görüntü seçin
5. "Toplu Analiz Et" butonuna tıklayın

### Vejetasyon Analizi
1. Harita sayfasına gidin (`/map`)
2. GeoTIFF dosyasını yükleyin
3. Vejetasyon indeksini seçin (NDVI, EVI, SAVI, vb.)
4. Renk haritasını seçin
5. Değer aralığını ayarlayın
6. "Analiz Et" butonuna tıklayın

### Proje Yönetimi
1. Projeler sayfasına gidin (`/projects`)
2. "Add New Farm" butonuna tıklayın
3. Farm, Field, Title, State bilgilerini girin
4. Proje fotoğrafını yükleyin
5. Projeyi kaydedin

## 📁 Proje Yapısı

```
farmvision/
├── detection/                 # Ana meyve tespit uygulaması
│   ├── views.py              # Tespit view fonksiyonları
│   ├── urls.py               # URL yapılandırması
│   ├── models.py             # Database modelleri
│   ├── templates/            # HTML şablonları
│   └── yolo/                 # YOLO tespit scriptleri
│       ├── detectcount.py    # Saymalı tespit
│       ├── detect.py         # Standart tespit
│       └── models/           # YOLO model yapıları
├── dron_map/                 # Harita ve coğrafi analiz
│   ├── views.py              # Harita işleme
│   └── models.py             # Proje modelleri
├── user_registration/        # Kullanıcı yönetimi
│   ├── views.py              # Auth işlemleri
│   └── models.py             # Kullanıcı modelleri
├── yolowebapp2/              # Ana Django uygulaması
│   ├── predict_tree.py       # Tespit fonksiyonları
│   ├── histogram.py          # Vejetasyon analizi
│   ├── hashing.py           # Dosya işleme
│   ├── options.py           # UI seçenekleri
│   ├── tasknode.py          # OpenDroneMap entegrasyonu
│   └── settings.py          # Django ayarları
├── static/                   # Statik dosyalar
│   ├── images/              # Yüklenen görüntüler
│   ├── images_counting/     # Çoklu tespit
│   ├── results/             # Analiz sonuçları
│   └── detected/            # İşlenmiş görüntüler
├── media/                    # Medya dosyaları
├── templates/                # Global şablonlar
├── docker-compose.yml        # Docker Compose konfigürasyonu
├── Dockerfile               # Docker imaj tanımı
├── requirements.txt          # Python bağımlılıkları
├── manage.py                # Django yönetim scripti
├── *.pt                     # YOLO model dosyaları
├── train.py                 # Model eğitim scripti
└── test.py                  # Model test scripti
```

## 🎮 API Endpoints

### Meyve Tespiti
- `GET /` - Ana tespit sayfası
- `POST /` - Tek görüntü analizi
- `GET /mcti` - Çoklu analiz sayfası
- `POST /mcti` - Çoklu görüntü analizi
- `GET /download_image/<slug>` - Sonuç indirme

### Harita ve Projeler
- `GET /map` - Harita arayüzü
- `GET /projects` - Proje listesi
- `GET /add_projects` - Yeni proje ekleme

### Kullanıcı Yönetimi
- `GET /login` - Giriş sayfası
- `POST /login_view` - Giriş işlemi
- `GET /signup` - Kayıt sayfası
- `GET /user_pr/<id>` - Kullanıcı profili
- `GET /user_edit/<id>` - Profil düzenleme
- `GET /calendar` - Takvim görünümü
- `GET /pricing` - Fiyatlandırma sayfası
- `GET /logout` - Çıkış işlemi

## ⚙️ Yapılandırma

### Meyve Ağırlıkları
```python
FRUIT_WEIGHTS = {
    'mandalina': 0.125,
    'elma': 0.105,
    'armut': 0.220,
    'seftale': 0.185,
    'nar': 0.300,
    'hurma': 0.050
}
```

### Vejetasyon İndeksi Parametreleri
```python
VEGETATION_INDICES = {
    'ndvi': '(NIR - Red) / (NIR + Red)',
    'evi': '2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)',
    'savi': '(1.5 * (NIR - Red)) / (NIR + Red + 0.5)',
    'gli': '(2*Green - Red - Blue) / (2*Green + Red + Blue)'
}
```

### Renk Haritası Seçenekleri
```python
COLORMAPS = ['rdylgn', 'spectral', 'viridis', 'plasma', 'jet', 'terrain']
```

### Model Parametreleri
```python
CONFIDENCE_THRESHOLD = 0.1
IMAGE_SIZE = 640
IOU_THRESHOLD = 0.45
```

## 🚀 Deployment

### 🐳 Docker Deployment

```bash
# Production deployment
docker-compose up -d

# Development ile logs
docker-compose up --build

# Container'lara erişim
docker exec -it murad bash
docker exec -it nodeodm bash
```

### 🔧 Manual Deployment

```bash
# Production sunucu için
gunicorn --bind 0.0.0.0:8000 yolowebapp2.wsgi:application

# Nginx ile reverse proxy
sudo apt install nginx
# /etc/nginx/sites-available/farmvision konfigürasyonu
```

### CLI Araçları
```bash
# Yeni model eğitimi
python train.py --data data/custom.yaml --cfg cfg/yolov7.yaml --weights yolov7.pt

# Model testi
python test.py --data data/custom.yaml --weights best.pt

# Standalone tespit
python detect.py --weights mandalina.pt --source test_images/

# Django management
python manage.py migrate
python manage.py collectstatic
python manage.py createsuperuser
```

## 📊 Özellikler Detayı

### Dashboard ve İstatistikler
- **İstatistik kartları** - Toplam meyve sayısı, ağırlık, işlem süresi
- **Grafik görselleştirme** - Chart.js ile Türkiye ağaç istatistikleri
- **Sidebar navigasyon** - Dashboard, Multi Counting, Calendar, Pricing, Projects
- **Profil yönetimi** - Kullanıcı bilgileri, avatar, rating sistemi

### Harita Arayüzü
- **Çoklu katman desteği** - Google Satellite, Hybrid, OSM
- **GeoTIFF görselleştirme** - Drone görüntülerini harita üzerinde görüntüleme
- **Koordinat görüntüleme** - Tıklanan noktanın lat/lng bilgisi
- **Zoom ve pan** - Harita navigasyon kontrolleri

### Kullanıcı Profili
- **Detaylı profil bilgileri** - İsim, soyisim, baba adı, doğum tarihi
- **İş bilgileri** - Departman, şehir, ülke bilgileri
- **İletişim** - Email, telefon, website bilgileri
- **Avatar sistemi** - Profil fotoğrafı yükleme ve düzenleme
- **Rating sistemi** - 5 yıldızlı kullanıcı değerlendirmesi

### UI/UX Detayları
- **Ana menü** - Dashboard, Multi Counting, Calendar, Pricing, Projects
- **Vejetasyon analizi** - Tab-based interface (Image Details + Analysis)
- **Range slider** - NDVI aralık seçimi için interaktif slider
- **Algorithm selector** - 20+ vejetasyon indeksi dropdown menüsü
- **Color picker** - 15+ renk paleti seçenekleri
- **Form validation** - Real-time hata mesajları ve uyarılar
- **Theme switcher** - 10 farklı renk teması (Red, Blue, Green, vb.)

### Sonuç İndirme ve Arşivleme
- **ZIP arşivleme** - Çoklu tespit sonuçlarını tek dosyada
- **Excel çıktısı** - Tespit sonuçlarının tablolu formatı
- **İşlenmiş görüntüler** - Bounding box'lı tespit sonuçları
- **Before/After görünümü** - Orijinal ve işlenmiş görüntülerin yan yana gösterimi

### Güvenlik ve Kullanıcı Yönetimi
- **CSRF koruması** - Tüm formlarda token doğrulaması
- **Authentication required** - Giriş yapmadan erişim engeli
- **Form validation** - Email, password, username doğrulaması
- **Error handling** - Güvenli hata mesajları
- **Session management** - Güvenli oturum yönetimi

## 📈 Optimizasyon

### Performans
- GPU kullanımı için CUDA desteği
- Batch processing için çoklu görüntü desteği
- Memory-efficient inference
- Real-time processing
- **Responsive Design** - Tüm cihazlarda uyumlu arayüz
- **Async Processing** - Non-blocking dosya yükleme
- **Interactive Maps** - 60fps smooth harita deneyimi
- **Lazy Loading** - Görüntülerin dinamik yüklenmesi
- **Theme Caching** - Kullanıcı tema tercihlerinin saklanması

### Desteklenen Formatlar
- **Görüntü**: JPG, PNG, BMP, TIFF
- **Coğrafi**: GeoTIFF (çok bantlı)
- **Çıktı**: Excel (XLSX), ZIP arşivi
- **Vejetasyon**: 4-bantlı raster (RGB + NIR)

## 🤝 Katkıda Bulunma

1. Repository'i fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

### Geliştirme Kuralları
- PEP 8 kod standartlarına uyun
- Docstring'leri ekleyin
- Unit testler yazın
- Model performansını test edin

## 📋 Roadmap

### v1.1 (Yakında)
- [ ] Video tespit desteği
- [ ] Kubernetes deployment
- [ ] Redis cache entegrasyonu
- [ ] Batch API endpoint'leri

### v1.2 (Gelecek)
- [ ] Multi-GPU desteği
- [ ] PostgreSQL production setup
- [ ] Nginx load balancer
- [ ] Celery task queue

### v2.0 (Uzun Vadeli)
- [ ] IoT sensör entegrasyonu
- [ ] Bulut tabanlı processing
- [ ] Enterprise multi-tenant desteği
- [ ] RESTful API v2

## 🐛 Bilinen Sorunlar

- **Docker Volumes**: Persistent storage sorunları olabilir
- **Model Dosyaları**: .pt dosyaları Git'e dahil değil, manuel indirme gerekli
- **Python Path**: predict_tree.py'de sabit kodlanmış Python yolu (`/myprojects/myprojectenv/bin/python3`)
- **GDAL Version**: GDAL 3.4.1 sürüm uyumluluğu kontrol edilmeli
- **Memory Management**: Büyük raster dosyalarda memory overflow

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👥 Takım

- **Backend Development**: Django + YOLOv7 + GDAL entegrasyonu
- **AI/ML**: YOLOv7 model optimizasyonu ve vejetasyon analizi
- **GIS Development**: Coğrafi veri işleme ve raster analizi
- **Frontend**: Web arayüzü ve harita görselleştirme

## 📞 İletişim

- **Repository**: [github.com/skyfarmx/farmvision](https://github.com/skyfarmx/farmvision)
- **Issues**: GitHub Issues sayfasını kullanın
- **Email**: info@skyfarmx.com

## 🙏 Teşekkürler

- YOLOv7 ekibine object detection modeli için
- Django topluluğuna web framework için
- GDAL/OGR geliştiricilerine geospatial kütüphaneler için
- Rasterio ve Rio-tiler ekiplerine raster işleme için
- OpenCV geliştiricilerine computer vision kütüphanesi için
- Beta test kullanıcılarımıza tarım sektöründen geri bildirimler için

---

**FarmVision ile tarımda AI ve uzaktan algılamanın gücünü keşfedin! 🌱🛰️🤖**
