"""
Kaydedilen NumPy dizilerini yükler.
EfficientNetB0 mimarisini kullanarak bir model oluşturur ve eğitir.
En iyi modeli best_model.keras dosyasına kaydeder.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# --- AYARLAR ---
IMG_SIZE   = 224
BATCH_SIZE = 8
LABEL_NAMES = ['F-16(A)', 'F-35(A)', 'TB2(A)', 'MiG-29(E)', 'Su-57(E)', 'Heron(E)', 'Civil(C)']

# --- VERİYİ YÜKLEME ---
print("Veri yükleniyor...")
X_train = np.load('numpy-set/X_train.npy')
X_test  = np.load('numpy-set/X_test.npy')
y_train = np.load('numpy-set/y_train.npy')
y_test  = np.load('numpy-set/y_test.npy')
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# --- AUGMENTATION ---
# Her sınıf için dengeli bir şekilde artırma yapacak şekilde ImageDataGenerator'ı yapılandırıyoruz. (Train setindeki sınıf dağılımına göre)
datagen = ImageDataGenerator(
    rotation_range=25,
    horizontal_flip=True,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.75, 1.25],
    shear_range=5,
    fill_mode='constant',
    cval=255
)

# --- MODEL ---
def build_model():
    base = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = True
    for layer in base.layers[:-80]:
        layer.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(7, activation='softmax')(x)

    model = Model(base.input, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()
print(f"Eğitilebilir katman: {sum(1 for l in model.layers if l.trainable)}")

# --- CALLBACKS ---
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
]

# --- EĞİTİM ---
print("\nEğitim başlıyor...")
# Sınıf ağırlıklarını hesapla
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Sınıf ağırlıkları:", class_weight_dict)

# model.fit() içine ekle:
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    class_weight=class_weight_dict,  # ← bunu ekle
    verbose=1
)
# --- SONUÇLAR ---
print("\n--- SONUÇLAR ---")
best = tf.keras.models.load_model('model/best_model.keras')

y_pred       = np.argmax(best.predict(X_test,  verbose=0), axis=1)
y_pred_train = np.argmax(best.predict(X_train, verbose=0), axis=1)

print(f"Train : %{accuracy_score(y_train, y_pred_train)*100:.1f}")
print(f"Test  : %{accuracy_score(y_test,  y_pred)*100:.1f}")
print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

# --- GRAFİK ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()

plt.savefig('training_curve.png', dpi=100)
print("\nGrafik kaydedildi: training_curve.png")