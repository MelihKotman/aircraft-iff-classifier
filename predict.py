import numpy as np
import tensorflow as tf
from PIL import Image
import sys

# --- AYARLAR ---
IMG_SIZE   = 224
CONFIDENCE_THRESHOLD = 0.70  # %70 altı → tanımlanamadı

LABEL_NAMES = ['F-16', 'F-35', 'TB2', 'MiG-29', 'Su-57', 'Heron', 'Civil']
IFF_MAP = {
    0: ('F-16',   'DOST', 'NATO üyesi, Batı menşeli fighter'),
    1: ('F-35',   'DOST', 'Stealth multirole fighter'),
    2: ('TB2',    'DOST', 'Türk yapımı taarruz drone'),
    3: ('MiG-29', 'DÜŞMAN', 'Sovyet/Rus menşeli fighter'),
    4: ('Su-57',  'DÜŞMAN', 'Rus 5. nesil stealth fighter'),
    5: ('Heron',  'DÜŞMAN', 'İsrail yapımı keşif drone'),
    6: ('Civil',  'BELİRSİZ', 'Sivil hava aracı')
}

# --- MODEL YÜKLE ---
print("Model yükleniyor...")
model = tf.keras.models.load_model('model/best_model.keras')

def load_image(path):
    img = Image.open(path).convert('RGBA')
    bg  = Image.new('RGBA', img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    img_rgb = bg.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img_rgb, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

def predict(image_path):
    print(f"\n{'='*50}")
    print(f"Görsel: {image_path}")
    print(f"{'='*50}")

    try:
        x = load_image(image_path)
    except Exception as e:
        print(f"Görsel yüklenemedi: {e}")
        return

    proba = model.predict(x, verbose=0)[0]
    pred  = np.argmax(proba)
    confidence = proba[pred] * 100

    print("\nSınıf olasılıkları:")
    for i, (name, prob) in enumerate(zip(LABEL_NAMES, proba)):
        bar = '█' * int(prob * 20)
        marker = ' ◄' if i == pred else ''
        print(f"  {name:<10} %{prob*100:5.1f}  {bar}{marker}")

    print(f"\nGüven skoru: %{confidence:.1f}")

    if confidence < CONFIDENCE_THRESHOLD * 100:
        print("\n SONUÇ   : TANIMLANAMADI")
        print("   DURUM   : SİVİL veya BİLİNMEYEN HAVA ARACI")
        print("   ÖNERİ   : Manuel doğrulama gerekli")
    else:
        model_name, iff_status, description = IFF_MAP[pred]
        print(f"\n MODEL   : {model_name}")
        print(f"   IFF     : {iff_status}")
        print(f"   BİLGİ   : {description}")

# --- KULLANIM ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python predict.py <görsel_yolu>")
        print("Örnek   : python predict.py test_ucak.png")
    else:
        predict(sys.argv[1])