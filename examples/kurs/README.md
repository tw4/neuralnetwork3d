# Kurs örnekleri + nn3d

`yapayzekakursu/ornekler` klasöründeki dört notebook'un kopyası. Mimariler ve
anlatım aynen korundu; her birine **nn3d canlı 3D görselleştirmesi** eklendi.

| Notebook | Mimari | nn3d'de görülecek şey |
|---|---|---|
| `fnn_musteri_kaybi_tahmini.ipynb` | Dense 128→64→32→1 | Özellik adları (`tenure`, `MonthlyCharges`…) aktivasyona göre renklenir |
| `cnn_moda_siniflandirma.ipynb` | 3 Conv bloğu, 16 katman | Konv katmanları **kanal** başına bir nokta; ağırlıksız katmanlar düz çizgi |
| `rnn_hisse_fiyat_tahmini.ipynb` | SimpleRNN + LSTM | Kendine dönen bağlantılar **mor halkalar** olarak |
| `gan_giysi_uretimi.ipynb` | DCGAN Generator (9.1M param) | Gürültü → resim genişlemesi: `7x7x512` → `14x14x256` → `28x28x3` |

## Çalıştırma

```bash
cd examples/kurs
jupyter lab
```

nn3d pip ile kurulu değilse notebook'lar depodaki `src/` klasörünü otomatik
bulur — ekstra kurulum gerekmez.

## Ne eklendi

Üçünde `nn3d.Monitor` bir Keras callback'i olarak `fit()`'e eklendi:

```python
izleyici = nn3d.Monitor(X_test[:1], every=10, input_labels=..., output_labels=...)
model.fit(..., callbacks=[early_stop, izleyici])
```

**GAN farklı**: kendi eğitim döngüsü var, `model.fit()` çağırmıyor. Orada
`nn3d.show()` ile görünüm açılıp döngünün içinden `gorunum.update(...)` ile her
epoch kare gönderiliyor.

## Orijinallerden farklar

Kopyalarken üç şey değişti — üçü de notebook'ları çalışır hâle getirmek için:

1. **`fnn`: pandas 3 zincirli atama hatası düzeltildi.**
   `df_clean['TotalCharges'].fillna(..., inplace=True)` pandas 3'te **sessizce
   hiçbir şey yapmıyor** (kopyanın üzerinde çalışıyor). Sonuç: 11 NaN kalıyor,
   eğitim NaN loss'a gidiyor. Doğrudan sütuna atamaya çevrildi. Bu hata kaynak
   notebook'ta da var; orayı da düzeltmek isteyebilirsin.

2. **`fnn`: CSV yolu.** Orijinal `'../telco_customer_churn.csv'` diyordu, bu
   depodan çözülmüyor. Artık birkaç aday yolu sırayla deniyor ve bulamazsa
   nereye baktığını söyleyerek hata veriyor.

3. **Hücre çıktıları temizlendi** — kayıtlı grafikler megabaytlarca base64 PNG
   demekti.

Model mimarileri, hiperparametreler ve anlatım metinleri **değiştirilmedi**.
