"""
GAN — Renkli Giysi Tasarımı Üretimi
=====================================
Fashion MNIST veri setindeki giysileri öğrenerek
tamamen yeni, renkli giysi tasarımları üretir.

Çalıştırmak için:
    python gan_giysi_uretimi.py

Paketleri kurmak için önce kurulum.bat dosyasını çalıştır.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ─── Tekrarlanabilirlik ───────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("  GAN — Renkli Giysi Üretimi Başlıyor")
print("=" * 60)
print(f"  TensorFlow : {tf.__version__}")
print(f"  GPU var mı : {len(tf.config.list_physical_devices('GPU')) > 0}")
print()

# ─── Sabitler ────────────────────────────────────────────────────────────────
LATENT_DIM  = 256    # Gürültü vektörü boyutu
BATCH_SIZE  = 128    # Her adımda işlenecek resim sayısı
EPOCHS      = 100    # Eğitim turu sayısı
GOSTER_HER  = 10     # Kaç epoch'ta bir görsel kaydedilsin

KAYIT_KLASORU = "gan_ciktilar"
os.makedirs(KAYIT_KLASORU, exist_ok=True)

CLASS_NAMES = [
    "T-Shirt", "Pantolon", "Kazak", "Elbise", "Mont",
    "Sandalet", "Gömlek", "Spor Ayakkabı", "Çanta", "Bot",
]

# Her kategori için RGB renk [R, G, B] — 0.0 ile 1.0 arası
CLASS_COLORS = np.array([
    [0.95, 0.20, 0.20],   # T-Shirt        → Kırmızı
    [0.15, 0.25, 0.80],   # Pantolon       → Koyu Mavi
    [0.95, 0.55, 0.10],   # Kazak          → Turuncu
    [0.80, 0.20, 0.75],   # Elbise         → Mor
    [0.15, 0.65, 0.25],   # Mont           → Yeşil
    [0.92, 0.78, 0.10],   # Sandalet       → Altın Sarısı
    [0.25, 0.70, 0.95],   # Gömlek         → Açık Mavi
    [0.95, 0.40, 0.10],   # Spor Ayakkabı  → Turuncu-Kırmızı
    [0.65, 0.25, 0.88],   # Çanta          → Violet
    [0.50, 0.28, 0.12],   # Bot            → Kahverengi
], dtype=np.float32)


# ─── 1. VERİ ─────────────────────────────────────────────────────────────────
def colorize(images_gray, labels):
    """
    Gri resimleri kategoriye göre renklendirir.
    Gri piksel × [R, G, B] = Renkli piksel
    """
    colored = np.zeros((len(images_gray), 28, 28, 3), dtype=np.float32)
    for i in range(len(images_gray)):
        renk = CLASS_COLORS[labels[i]]
        colored[i] = images_gray[i, :, :, None] * renk
    return colored


print("[1/5] Veri yükleniyor ve renkli hale getiriliyor...")
(X_raw, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()

X_gray  = X_raw.astype(np.float32) / 255.0          # [0,255] → [0,1]
X_color = colorize(X_gray, y_train)                  # Renklendirme
X_color = (X_color * 2.0) - 1.0                      # [0,1] → [-1,+1]

dataset = (
    tf.data.Dataset.from_tensor_slices(X_color)
    .shuffle(60000)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

print(f"    Veri şekli  : {X_color.shape}")
print(f"    Değer aralığı: [{X_color.min():.1f}, {X_color.max():.1f}]")
print(f"    Batch sayısı : {len(dataset)}")
print()


# ─── 2. MODEL: GENERATOR ─────────────────────────────────────────────────────
def build_generator():
    """
    Rastgele gürültüden (256,) renkli resim (28,28,3) üretir.

    Mimari:
        Gürültü (256,)
        → Dense → (7×7×512)          ← Düzleştirilmiş özellik haritası
        → Conv2DTranspose (stride=2) → (14×14×256)   ← 2× büyüt
        → Conv2DTranspose (stride=2) → (28×28×128)   ← 2× büyüt
        → Conv2DTranspose (stride=1) → (28×28×3)     ← RGB çıkış, tanh ile [-1,+1]
    """
    return models.Sequential([
        # Gürültü → 7×7×512 özellik haritası
        layers.Dense(7 * 7 * 512, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((7, 7, 512)),

        # 7×7 → 14×14
        layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        # 14×14 → 28×28
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        # Son katman — RGB çıkış
        layers.Conv2DTranspose(3, (4, 4), strides=(1, 1), padding="same",
                               use_bias=False, activation="tanh"),
    ], name="Generator")


# ─── 3. MODEL: DISCRIMINATOR ─────────────────────────────────────────────────
def build_discriminator():
    """
    Gelen resmin gerçek mi sahte mi olduğunu ayırt eder.

    Mimari:
        Resim (28,28,3)
        → Conv2D (stride=2) → (14×14×64)
        → Conv2D (stride=2) → (7×7×128)
        → Conv2D (stride=2) → (4×4×256)
        → Flatten → Dense(1)   ← Tek sayı: pozitif=gerçek, negatif=sahte
    """
    return models.Sequential([
        layers.Conv2D(64,  (4, 4), strides=(2, 2), padding="same", input_shape=(28, 28, 3)),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(256, (4, 4), strides=(2, 2), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dense(1),   # Aktivasyon yok — from_logits=True kullanılıyor
    ], name="Discriminator")


print("[2/5] Modeller oluşturuluyor...")
generator     = build_generator()
discriminator = build_discriminator()
print(f"    Generator    : {generator.count_params():,} parametre")
print(f"    Discriminator: {discriminator.count_params():,} parametre")
print()


# ─── 4. EĞİTİM AYARLARI ──────────────────────────────────────────────────────
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(sahte_cikis):
    # Generator, Discriminator'ın sahteye "gerçek (1)" demesini istiyor
    return bce(tf.ones_like(sahte_cikis), sahte_cikis)


def discriminator_loss(gercek_cikis, sahte_cikis):
    # Gerçekleri 1, sahteleri 0 olarak tanımalı
    # 0.9: label smoothing — Discriminator aşırı güvenmesin
    gercek_kayip = bce(tf.ones_like(gercek_cikis) * 0.9, gercek_cikis)
    sahte_kayip  = bce(tf.zeros_like(sahte_cikis),         sahte_cikis)
    return gercek_kayip + sahte_kayip


gen_optimizer  = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)


@tf.function
def egitim_adimi(gercek_resimler):
    """Tek batch için eğitim adımı. @tf.function ile 3-5× hızlandırılmış."""
    gurultu = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        sahte_resimler = generator(gurultu, training=True)
        gercek_cikis   = discriminator(gercek_resimler, training=True)
        sahte_cikis    = discriminator(sahte_resimler,  training=True)
        g_kayip        = generator_loss(sahte_cikis)
        d_kayip        = discriminator_loss(gercek_cikis, sahte_cikis)

    gen_optimizer.apply_gradients(
        zip(gen_tape.gradient(g_kayip,  generator.trainable_variables),
            generator.trainable_variables)
    )
    disc_optimizer.apply_gradients(
        zip(disc_tape.gradient(d_kayip, discriminator.trainable_variables),
            discriminator.trainable_variables)
    )
    return g_kayip, d_kayip


# ─── 5. EĞİTİM ───────────────────────────────────────────────────────────────
SABIT_GURULTU = tf.random.normal([20, LATENT_DIM])   # İzleme için sabit vektörler


def gorsel_kaydet(epoch, gen_hist, disc_hist):
    """Üretilen resimleri ve kayıp grafiğini PNG olarak kaydeder."""
    tahminler = generator(SABIT_GURULTU, training=False).numpy()
    tahminler = np.clip((tahminler + 1) / 2, 0, 1)

    fig = plt.figure(figsize=(22, 6))

    # 20 üretilen resim
    for i in range(20):
        ax = fig.add_subplot(2, 13, i + 1 if i < 10 else i + 4)
        ax.imshow(tahminler[i])
        ax.axis("off")

    # Kayıp grafiği
    ax_k = fig.add_subplot(1, 4, 4)
    ax_k.plot(gen_hist,  color="#2196F3", linewidth=2, label="Generator")
    ax_k.plot(disc_hist, color="#FF5722", linewidth=2, linestyle="--", label="Discriminator")
    ax_k.axhline(0.693, color="gray", linestyle=":", alpha=0.7, label="İdeal ~0.69")
    ax_k.set_title("Kayıp Geçmişi")
    ax_k.set_xlabel("Epoch")
    ax_k.legend(fontsize=8)
    ax_k.grid(True, alpha=0.3)

    fig.suptitle(f"Epoch {epoch} — GAN Renkli Giysiler", fontsize=13, fontweight="bold")
    plt.tight_layout()

    dosya = os.path.join(KAYIT_KLASORU, f"epoch_{epoch:03d}.png")
    plt.savefig(dosya, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    Görsel kaydedildi → {dosya}")


print("[3/5] Eğitim başlıyor...")
print(f"    {EPOCHS} epoch × {len(dataset)} batch = {EPOCHS * len(dataset)} adım")
print("=" * 60)

gen_hist, disc_hist = [], []
baslangic = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_gen, epoch_disc = [], []

    for batch in dataset:
        g_k, d_k = egitim_adimi(batch)
        epoch_gen.append(float(g_k))
        epoch_disc.append(float(d_k))

    gen_hist.append(np.mean(epoch_gen))
    disc_hist.append(np.mean(epoch_disc))

    gecen = (time.time() - baslangic) / 60
    print(
        f"  Epoch {epoch:3d}/{EPOCHS} | "
        f"Gen: {gen_hist[-1]:.4f} | "
        f"Disc: {disc_hist[-1]:.4f} | "
        f"{gecen:.1f} dk"
    )

    if epoch % GOSTER_HER == 0 or epoch == 1:
        gorsel_kaydet(epoch, gen_hist, disc_hist)

toplam_sure = (time.time() - baslangic) / 60
print()
print(f"[4/5] Eğitim tamamlandı! Toplam süre: {toplam_sure:.1f} dakika")


# ─── 6. FİNAL GÖRSELLER ──────────────────────────────────────────────────────
print("[5/5] Final görseller üretiliyor...")

# 80 yeni giysi
gurultu  = tf.random.normal([80, LATENT_DIM])
uretilen = generator(gurultu, training=False).numpy()
uretilen = np.clip((uretilen + 1) / 2, 0, 1)

fig, axes = plt.subplots(8, 10, figsize=(20, 16))
for i, ax in enumerate(axes.flat):
    ax.imshow(uretilen[i])
    ax.axis("off")
plt.suptitle(f"GAN — 80 Yeni Renkli Giysi Tasarımı ({EPOCHS} Epoch)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
final_dosya = os.path.join(KAYIT_KLASORU, "final_80_giysi.png")
plt.savefig(final_dosya, dpi=120, bbox_inches="tight")
plt.close()
print(f"    80 giysi görseli → {final_dosya}")

# Kayıp grafiği
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, EPOCHS + 1), gen_hist,  color="#2196F3", linewidth=2, label="Generator Kaybı")
ax.plot(range(1, EPOCHS + 1), disc_hist, color="#FF5722", linewidth=2, linestyle="--", label="Discriminator Kaybı")
ax.axhline(0.693, color="gray", linestyle=":", alpha=0.7, label="İdeal denge ~0.69")
ax.set_title("GAN Eğitim Kaybı", fontsize=13, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Kayıp")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
kayip_dosya = os.path.join(KAYIT_KLASORU, "egitim_kaybi.png")
plt.savefig(kayip_dosya, dpi=120, bbox_inches="tight")
plt.close()
print(f"    Kayıp grafiği    → {kayip_dosya}")

# Modeli kaydet
model_dosya = "gan_renkli_generator.keras"
generator.save(model_dosya)
print(f"    Model kaydedildi → {model_dosya}")

print()
print("=" * 60)
print("  Tüm çıktılar 'gan_ciktilar/' klasörüne kaydedildi.")
print("  Modeli tekrar kullanmak için:")
print("    gen = tf.keras.models.load_model('gan_renkli_generator.keras')")
print("    img = gen(tf.random.normal([1, 256]), training=False)")
print("=" * 60)
