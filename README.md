# 🚁 Farm Vision  
AI-Powered Tarımsal Drone Görüntü Analizi Platformu

## 🌾 Platform Hakkında
Farm Vision, drone teknolojisi ve yapay zekayı birleştirerek modern tarım ihtiyaçlarına yönelik kapsamlı bir web platformudur. Çiftçiler, tarım uzmanları ve araştırmacılar için tasarlanmış olan bu platform, drone ile çekilen tarımsal görüntüleri analiz ederek değerli bilgiler sunar.

---

## ✨ Temel Özellikler

### 🔍 Akıllı Görüntü Analizi
- **Bitki Sağlığı Tespiti:** AI algoritmaları ile bitki sağlık durumu analizi  
- **Hastalık ve Zararlı Tespiti:** Erken teşhis ile zamanında müdahale  
- **Verim Tahminleme:** Hasat öncesi verim projeksiyonları  
- **Su Stresi Analizi:** Sulama ihtiyaçlarının belirlenmesi  

### 📊 Veri Yönetimi ve Raporlama
- İnteraktif haritalar ve görsel analizler  
- Zaman serisi takibi  
- Karşılaştırmalı dönemsel raporlar  
- PDF, Excel ve diğer formatlarda rapor indirme  

### 🎯 Hassas Tarım Çözümleri
- Bölgesel müdahale haritaları  
- Kaynak optimizasyonu (gübre, ilaç, su)  
- Maliyet analizi ve sürdürülebilir tarım destekleri  

---

## 🚀 Teknoloji Altyapısı

### Backend & Database
- Django Framework  
- PostgreSQL  
- Redis  

### AI & Machine Learning
- PyTorch  
- YOLOv8  
- OpenCV  
- scikit-learn  

### Görselleştirme & Analiz
- TensorBoard  
- Pandas & NumPy  
- Matplotlib & Seaborn  

---

## 📋 Sistem Gereksinimleri

| Bileşen   | Minimum |
|-----------|---------|
| CPU       | 4 çekirdek (Intel i5 / Ryzen 5) |
| RAM       | 8 GB DDR4 |
| Depolama  | 50 GB SSD |
| GPU       | NVIDIA GTX 1060 veya üzeri (önerilen) |

**Desteklenen Sistemler:** Ubuntu 20.04+, CentOS 8+, Docker

---

## 🛠️ Kurulum

### 1. Sistem Hazırlığı
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.10 python3.10-dev python3.10-venv postgresql postgresql-contrib redis-server libopencv-dev python3-opencv
````

### 2. Proje Kurulumu

```bash
git clone https://github.com/yourcompany/farm-vision.git
cd farm-vision
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-ubuntu22.txt
```

### 3. Veritabanı Konfigürasyonu

```bash
sudo -u postgres createdb farm_vision_db
sudo -u postgres createuser farm_vision_user
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

### 4. Servisi Başlatma

* Development:

```bash
python manage.py runserver
```

* Production:

```bash
gunicorn farm_vision.wsgi:application --bind 0.0.0.0:8000
```

---

## 📱 Kullanım

* **Admin Panel:** `http://localhost:8000/admin/` adresinden erişim
* **Drone Görüntüsü Yükleme:** Yeni proje oluşturup, görüntüleri yükleyin
* **AI Analizi:** Yüklenen görüntüler otomatik işlenir, sonuçlar harita üzerinde gösterilir
* **Raporlama:** PDF ve Excel formatında detaylı raporlar

---

## 📊 Desteklenen Analiz Türleri

| Analiz Türü      | Açıklama                            | Çıktı Formatı                        |
| ---------------- | ----------------------------------- | ------------------------------------ |
| NDVI Analizi     | Bitki sağlığı ve yeşillik indeksi   | Renkli harita, sayısal değer         |
| Hastalık Tespiti | Bitki hastalıklarının erken teşhisi | Marked areas, güven skorları         |
| Zararlı Analizi  | Böcek ve zararlı tespiti            | Konum işaretleri, yoğunluk haritası  |
| Su Stresi        | Sulama ihtiyacı analizi             | Su stress haritası, öncelik alanları |
| Verim Tahmini    | Hasat öncesi verim projeksiyonu     | Ton/hektar, toplam verim             |
| Büyüme Analizi   | Bitki gelişim takibi                | Zaman serisi grafikleri              |

---

## 🔒 Güvenlik Özellikleri

* SSL/TLS şifrelemesi
* Django Security Middleware
* PostgreSQL güvenliği
* Kullanıcı doğrulama ve yetkilendirme
* API Rate Limiting
* GDPR uyumu

---

## 🤝 Destek ve İletişim

* Email: [support@skyfarmx.com](mailto:support@skyfarmx.com)
* Telefon: +90 (553) 309-1312
* Live Chat: Platform üzerinden 7/24

---

## 📄 Lisans

MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

---

## 🙏 Katkıda Bulunanlar

Farm Vision, tarım teknolojisi ve yapay zeka alanında uzman ekip tarafından geliştirilmiştir.

---

*Farm Vision - Tarımın Geleceğini Görmek 🌱✨*
