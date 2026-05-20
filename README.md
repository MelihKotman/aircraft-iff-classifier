# ✈️ Aircraft IFF Classifier

Görsel tabanlı bir **IFF (Identification Friend or Foe)** sistemi. Hava araçlarını fotoğraftan tanıyarak dost/düşman/sivil olarak sınıflandırır.

---

## 🎯 Proje Özeti

| | |
|---|---|
| **Model** | EfficientNetB3 (Transfer Learning) |
| **Test Accuracy** | %98.03 |
| **Sınıf Sayısı** | 7 (6 askeri + 1 sivil) |
| **Toplam Görsel** | ~335 |
| **Preprocessing** | CLAHE (düşük ışık dayanıklılığı) |

---

## 🛩️ Tanınan Sınıflar

| Sınıf | IFF | Açıklama |
|-------|-----|----------|
| F-16 | DOST ✅ | NATO üyesi, Batı menşeli fighter |
| F-35 | DOST ✅ | Stealth multirole fighter |
| TB2 | DOST ✅ | Türk yapımı taarruz drone |
| MiG-29 | DÜŞMAN ❌ | Sovyet/Rus menşeli fighter |
| Su-57 | DÜŞMAN ❌ | Rus 5. nesil stealth fighter |
| Heron | DÜŞMAN ❌ | Keşif drone |
| Civil | BELİRSİZ ⚠️ | Sivil hava aracı |

---

## 🏗️ Mimari

```
Görsel (224x224)
      ↓
CLAHE Preprocessing  ← Karanlık/sisli görüntü iyileştirme
      ↓
EfficientNetB3       ← ImageNet ağırlıkları (son 50 katman açık)
      ↓
GlobalAveragePooling
      ↓
Dense(128) → Dropout(0.5)
Dense(64)  → Dropout(0.3)
      ↓
Dense(7, softmax)    ← 7 sınıf
      ↓
Güven Eşiği (%70)   ← Altındaysa → TANIMLANAMADI
```

---

## 📊 Sonuçlar

```
              precision    recall  f1-score
F-16 (DOST)     1.00      0.89      0.94
F-35 (DOST)     1.00      1.00      1.00
TB2  (DOST)     1.00      1.00      1.00
MiG-29 (DÜŞMAN) 0.89      1.00      0.94
Su-57  (DÜŞMAN) 1.00      1.00      1.00
Heron  (DÜŞMAN) 1.00      1.00      1.00
Civil           1.00      1.00      1.00

accuracy                           0.98
```

---

## 🚀 Kurulum

```bash
git clone https://github.com/MelihKotman/aircraft-iff-classifier.git
cd aircraft-iff-classifier

pip install tensorflow pillow scikit-learn opencv-python numpy
```

---

## 💻 Kullanım

### 1. Veri setini hazırla

```
dataset/
├── ally/          ← f16_1.png, f35_1.png, tb2_1.png ...
├── enemy/         ← mig_1.png, su_1.png, heron_1.png ...
└── civil/         ← civil_1.png ...
```

### 2. Dataset oluştur

```bash
python build_dataset.py
```

### 3. Modeli eğit

```bash
python train_model.py
```

### 4. Tahmin yap

```bash
python predict.py dataset/test/test_1.png
```

**Örnek çıktı:**
```
==================================================
Görsel: dataset/test/test_1.png
==================================================

Sınıf olasılıkları:
  F-16        %  0.0
  F-35        %  0.0
  TB2         %  0.0
  MiG-29      %  1.0
  Su-57       % 97.7  ███████████████████ ◄
  Heron       %  0.0
  Civil       %  0.0

Güven skoru: %97.7

 MODEL   : Su-57
   IFF     : DÜŞMAN
   BİLGİ   : Rus 5. nesil stealth fighter
```

---

## 🔑 Temel Özellikler

**CLAHE Preprocessing** — Görüntüdeki parlaklık dengesizliğini giderir. Karanlık veya sisli koşullarda çekilen hava aracı fotoğraflarında model performansını korur.

**Güven Eşiği** — Model %70 altında güven üretirse "TANIMLANAMADI" uyarısı verir. Boeing 737 gibi tanımadığı sivil uçakları zorla sınıflandırmaz.

**Stratified Split** — Her sınıf train/test'e eşit oranda dağıtılır. Veri dengesizliği class weights ile telafi edilir.

---

## 📁 Dosya Yapısı

```
aircraft-iff-classifier/
├── build_dataset.py    ← Görselleri yükler, numpy olarak kaydeder
├── train_model.py      ← EfficientNetB0 eğitimi
├── predict.py          ← Tek görsel tahmini
├── best_model.keras    ← Eğitilmiş model (git-ignore)
└── dataset/
    ├── ally/
    ├── enemy/
    └── civil/
```

> ⚠️ `best_model.keras` ve `.npy` dosyaları boyut sınırı nedeniyle repoya dahil edilmemiştir. Modeli yeniden oluşturmak için sırasıyla `build_dataset.py` ve `train_model.py` çalıştırın.

---

## 🛠️ Teknolojiler

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-red)
