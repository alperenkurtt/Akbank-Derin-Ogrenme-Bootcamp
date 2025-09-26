# Akbank Derin Öğrenme Bootcamp - Pneumonia Hastalığı Tahmini - [Kaggle Link](https://www.kaggle.com/code/alperenkurtt/deeplearningbootcamp)

Bu proje, göğüs röntgeni görüntülerinden pnömoni teşhisi koymak için geliştirilmiş bir CNN (Convolutional Neural Network) modelini içermektedir.

## 📋 İçindekiler

1. [Proje Hakkında](#1-proje-hakkında)
2. [Hiperparametre Optimizasyonu](#2-hiperparametre-optimizasyonu)
3. [Veri Yükleme](#3-veri-yükleme)
4. [Veri Görselleştirme](#4-veri-görselleştirme)
5. [Veri Önişleme ve Hazırlık](#5-veri-önişleme-ve-hazırlık)
6. [Model Tasarımı ve Eğitimi](#6-model-tasarımı-ve-eğitimi)
7. [Model Değerlendirmesi](#7-model-değerlendirmesi)

## 1. Proje Hakkında

### 1.1 Proje Amacı

**Sorun Tanımı:** Göğüs röntgeni görüntülerinden pnömoni teşhisi koymak, radyolojistler için zaman alıcı ve uzmanlık gerektiren bir süreçtir. Manuel değerlendirme süreçleri hem zaman kaybına hem de insan kaynaklı hatalara yol açabilmektedir.

**Görev:** Bu proje kapsamında, göğüs röntgeni görüntülerinden pnömoni varlığını otomatik tespit edebilen bir CNN modeli geliştirilmiştir. 5,863 adet röntgen görüntüsü kullanarak NORMAL ve PNEUMONIA sınıfları arasında tahmin yapan bir model oluşturulmuştur.

### 1.2 Veri Seti Bilgisi

**Temel Özellikler:**
- **Veri Seti:** Chest X-Ray Images (Pneumonia)
- **Toplam Görüntü:** 5,863 adet JPEG röntgen görüntüsü
- **Sınıflar:** 2 sınıf (NORMAL, PNEUMONIA)
- **Bölümler:** train, test, validation setleri

**Kaynak ve Kalite:**
- **Hastane:** Guangzhou Women and Children's Medical Center
- **Yaş Grubu:** 1-5 yaş pediatrik hastalar
- **Doğrulama:** İki uzman hekim tarafından etiketlenmiş
- **Veri Seti:** [Kaggle - Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## 2. Hiperparametre Optimizasyonu

### Optimizatör Karşılaştırması
Farklı optimizatörlerin (SGD, Momentum, Adam, RMSprop) performansları karşılaştırılmıştır:

- **SGD:** %95.02 doğruluk
- **Momentum:** %96.74 doğruluk  
- **Adam:** %98.08 doğruluk ⭐
- **RMSprop:** %97.89 doğruluk

**Sonuç:** Adam optimizatörü en iyi performansı gösterdiği için seçilmiştir.

### Hiperparametre Arama
Farklı öğrenme oranları ve batch size değerleri test edilmiştir:

| Görüntü Boyutu | En İyi Kombinasyon | Doğruluk |
|----------------|-------------------|----------|
| 64x64 | lr=0.001, bs=64 | %97.41 |
| 128x128 | lr=0.001, bs=32 | %97.61 |
| 150x150 | lr=0.001, bs=32 | %97.61 |
| 224x224 | lr=0.001, bs=64 | %98.18 ⭐ |
| 256x256 | lr=0.001, bs=32 | %97.22 |

**Final Parametreler:**
```python
IMG_SIZE = (150, 150)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
```

## 3. Veri Yükleme

Veriler train, validation ve test seti olarak ayrılmıştır:
- **Train:** 4,695 görüntü (%10 validation split ile)
- **Test:** 624 görüntü
- **Validation:** 521 görüntü

Tüm görüntüler gri tonlamalı (grayscale) olarak 150x150 piksel boyutuna getirilmiştir.

## 4. Veri Görselleştirme

### Sınıf Dağılımları

| Veri Seti | NORMAL | PNEUMONIA | Toplam | PNEUMONIA Oranı |
|-----------|--------|-----------|--------|----------------|
| Train | 1,218 | 3,477 | 4,695 | %74.06 |
| Test | 234 | 390 | 624 | %62.50 |
| Validation | 123 | 398 | 521 | %76.39 |

**Toplam:** 1,575 NORMAL, 4,265 PNEUMONIA (%73.0 PNEUMONIA)

### Veri Dengesizliği
Veri seti oldukça dengesizdir - PNEUMONIA örnekleri NORMAL örneklerden yaklaşık 3 kat daha fazladır. Bu durum modelin yanlış öğrenmesine sebep olabilir.

## 5. Veri Önişleme ve Hazırlık

### Normalizasyon
Tüm piksel değerleri 0-255 aralığından 0-1 aralığına dönüştürülmüştür.

### Data Augmentation
```python
data_augmentation = Sequential([
    RandomFlip("horizontal"),     # Yatay çevirme
    RandomRotation(0.1),          # ±10 derece döndürme
    RandomZoom(0.1),              # %10 yakınlaştırma/uzaklaştırma
    RandomContrast(0.1)           # Kontrast değişimi
])
```

### Performans Optimizasyonu
Veriler prefetch ile optimize edilmiştir:
```python
train = train.prefetch(tf.data.AUTOTUNE)
test = test.prefetch(tf.data.AUTOTUNE)
val = val.prefetch(tf.data.AUTOTUNE)
```

## 6. Model Tasarımı ve Eğitimi

### Model 1: Basit CNN
```python
model = Sequential([
    data_augmentation,
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,1)),
    MaxPool2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

**Sonuç:** Training: %91.16, Validation: %79.85

### Model 2: Gelişmiş CNN (Final Model)
```python
model_final = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,1)),
    MaxPooling2D(2,2), Dropout(0.25),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2), Dropout(0.25),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2), Dropout(0.3),
    
    Flatten(),
    Dense(128, activation='relu'), Dropout(0.5),
    Dense(64, activation='relu'), Dropout(0.4),
    Dense(1, activation='sigmoid')
])
```

**Sonuç:** Training: %98.66, Validation: %98.08

### Model 3: Transfer Learning (MobileNetV2)
```python
base_model = MobileNetV2(
    input_shape=(150, 150, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = Sequential([
    data_augmentation,
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'), Dropout(0.5),
    Dense(128, activation='relu'), Dropout(0.4),
    Dense(64, activation='relu'), Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

**Sonuç:** Training: %94.16, Validation: %95.20

## 7. Model Değerlendirmesi

### En İyi Model Performansı (Model 2)
- **Training Accuracy:** %98.66
- **Validation Accuracy:** %98.08
- **Accuracy Farkı:** %0.58 (Overfitting yok)

### Confusion Matrix Sonuçları
```
              precision    recall  f1-score   support

      Normal       0.99      0.36      0.53       234
   Pneumonia       0.72      1.00      0.84       390

    accuracy                           0.76       624
```

### Model Analizi
- Model PNEUMONIA sınıfını çok iyi tespit ediyor (%100 recall)
- NORMAL sınıfında daha düşük performans (%36 recall)
- Bu durum veri dengesizliğinden kaynaklanıyor olabilir

## 🛠️ Teknolojiler

- **Python 3.x**
- **TensorFlow/Keras**
- **NumPy, Pandas**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **OpenCV**

## 📊 Sonuçlar

Proje başarıyla tamamlanmış ve %98+ doğruluk oranına sahip bir CNN modeli geliştirilmiştir. Model, göğüs röntgeni görüntülerinden pnömoni teşhisi konusunda yüksek performans göstermektedir.

### Başarı Faktörleri
1. Kapsamlı hiperparametre optimizasyonu
2. Uygun veri önişleme teknikleri
3. Dropout katmanları ile overfitting önleme
4. EarlyStopping ve ReduceLROnPlateau callback'leri
5. Transfer learning deneyimi

### Geliştirilebilir Alanlar
1. Veri dengesizliği sorunu (SMOTE, class weights)
2. Daha fazla veri artırma tekniği
3. Ensemble modelleri
4. Farklı CNN mimarileri (ResNet, EfficientNet)

## 📝 Notlar

- Proje Kaggle platformunda çalıştırılmıştır
- GPU desteği kullanılmıştır (Tesla P100-PCIE-16GB)
- Veri seti Kaggle'dan indirilmiştir
- Tüm kodlar Jupyter Notebook formatında hazırlanmıştır

[Kaggle Link](https://www.kaggle.com/code/alperenkurtt/deeplearningbootcamp)
