"""
Bu dosya ne yapıyor?
- Görselleri yükleyip, 224x224 boyutuna getiriyor.
- Her görselin hangi sınıf olduğunu dosya adından tahmin ediyor.
- %80 train, %20 test olarak bölüyor.
- X_train, y_train, X_test, y_test olarak kaydediyor.
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

# AYARLAR
IMG_SIZE = 224
DATASET_FOLDER = 'dataset'  # Görsellerin bulunduğu klasör

LABEL_MAP = {
    'f16': 0,
    'f35': 1,
    'tb2': 2,
    'mig': 3,
    'su': 4,
    'heron': 5,
    'civil': 6
}
LABEL_NAMES = ['F-16', 'F-35', 'TB2', 'MiG-29', 'Su-57', 'Heron', 'Civil']

def get_subclass(filename):
    name = filename.lower()
    if 'f16' in name: return 'f16'
    if 'f35' in name: return 'f35'
    if 'tb2' in name: return 'tb2'
    if 'mig' in name: return 'mig'
    if 'su' in name: return 'su'
    if 'heron' in name: return 'heron'
    if 'civil' in name: return 'civil'
    return None

def load_image(path):
    """
    Şeffaf arka planı temizler, ardından karanlık/sisli görüntülerdeki
    uçak hatlarını belirginleştirmek için OpenCV CLAHE filtresi uygular.
    """
    # 1. PIL İLE ŞEFFAFLIK (ALPHA) TEMİZLİĞİ (Senin harika kodun)
    img = Image.open(path).convert('RGBA')
    bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    img_rgb = bg.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    
    # Görüntüyü OpenCV'nin anlayacağı 0-255 arası formatına çevir
    img_array = np.array(img_rgb, dtype=np.uint8)

    # 2. OPENCV İLE CLAHE EKLENTİSİ (Karanlık/Sis Aşma)
    # RGB görüntüyü LAB renk uzayına çeviriyoruz. Neden? 
    # Çünkü sadece 'L' (Luminance/Parlaklık) kanalına işlem yapacağız, renkleri(A,B) bozmayacağız.
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)

    # CLAHE objesini oluştur
    # clipLimit: Kontrastın dozudur (2.0 ile 3.0 arası idealdir, çok artarsa karıncalanma yapar)
    # tileGridSize: Görüntüyü 8x8 kutulara bölüp her kutuyu kendi içinde aydınlatır
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # İşlenmiş parlaklık kanalını (cl) orijinal renk kanallarıyla (a,b) geri birleştir
    merged = cv2.merge((cl, a, b))
    
    # LAB formatından tekrar Keras/Model için RGB'ye dönüştür
    final_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    # Derin öğrenme modelinin beklediği float32 formatına çevirip yolla
    return final_rgb.astype(np.float32)

# GÖRSELLERİ YÜKLEME

all_imges = []
all_labels = []
all_subclasses = []

print("Görseller yükleniyor...")
for class_name in ['ally','enemy','civil']:
    folder = os.path.join(DATASET_FOLDER, class_name)
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        subclass = get_subclass(fname)
        if subclass is None:
            print(f"Uyarı: '{fname}' dosyasından sınıf tahmin edilemedi, atlanıyor.")
            continue
        path = os.path.join(folder, fname)
        img_array = load_image(path)
        all_imges.append(img_array)
        all_labels.append(LABEL_MAP[subclass])
        all_subclasses.append(subclass)

X = np.array(all_imges)
y = np.array(all_labels)

print(f"Toplam {len(X)} görsel yüklendi.")
print("Sınıf dağılımı:")
for label, name in enumerate(LABEL_NAMES):
    count = np.sum(y == label)
    print(f"  {name}: {count} görsel")

# VERİYİ BÖLME 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=all_subclasses  # Her sınıf eşit oranda bölünsün
)

np.save('numpy-set/X_train.npy', X_train)
np.save('numpy-set/X_test.npy', X_test)
np.save('numpy-set/y_train.npy', y_train)
np.save('numpy-set/y_test.npy', y_test)

print("Veri başarıyla kaydedildi:")
print(f"  X_train.npy: {X_train.shape}")
print(f"  X_test.npy: {X_test.shape}")
print(f"  y_train.npy: {y_train.shape}")
print(f"  y_test.npy: {y_test.shape}")