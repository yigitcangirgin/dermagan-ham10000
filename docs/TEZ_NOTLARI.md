# Tez notları: tezden koda

Bu belge, bitirme projesi dokümanındaki (Haziran 2024) her adımın kodda nereye
karşılık geldiğini, kodda bilinçli olarak değiştirilen noktaları ve neden
değiştirildiklerini açıklar.

## Tezdeki adım → kod

| Tezdeki adım | Kod | Not |
| --- | --- | --- |
| HAM10000'in iki görsel klasörünü birleştirme, CSV'den etiketleme | `data.py` → `load_ham10000` | Eksik görseller atlanır ve uyarı verilir |
| %80 eğitim / %20 test, ardından %80 eğitim / %20 doğrulama | `data.py` → `split_by_lesion` | Lezyon düzeyinde ve sınıf dengeli (aşağıya bakın) |
| cGAN ile veri artırımı | `gan.py` → `Generator`, `Discriminator`, `train_gan`, `synthesize` | 64x64 koşullu DCGAN |
| Döndürme, kaydırma, ölçekleme, parlaklık, çevirme | `torch_data.py` → `classifier_transforms` | `ImageDataGenerator` yerine torchvision |
| VGG16, GoogLeNet (Inception v1), ResNet50, ImageNet ağırlıkları | `classifiers.py` → `build_model` | Son katman 7 sınıflı yeni katmanla değiştirilir |
| Adam, lr 0.001, batch 32, 20 epoch, categorical cross-entropy | `classifiers.py` → `train_cnn` | CLI varsayılanları bunlar |
| Erken durdurma, en iyi modeli kaydetme | `train_cnn` | Doğrulama kaybına göre, `patience=5` |
| GoogLeNet'ten özellik çıkarıp yeni sınıflandırıcı eğitme | `extract_features` + `ml.py` | RF (100 ağaç), KNN (k=5), Karar Ağacı |
| Accuracy, precision, recall, F1 (weighted), specificity | `metrics.py` | Ek olarak macro F1 ve ROC AUC |

## Bilinçli değişiklikler

1. **Lezyon düzeyinde bölme.** HAM10000'de aynı lezyonun birden fazla
   fotoğrafı var. Görsel bazında rastgele bölünce neredeyse aynı fotoğraflar hem
   eğitimde hem testte yer alıyor ve skorlar olduğundan yüksek çıkıyor. Bu
   nedenle aynı `lesion_id`'ye ait tüm görseller tek bir bölümde tutuluyor.
   Sonuçlar tezdekinden düşük çıkabilir; bu beklenen bir durum ve daha dürüst
   bir ölçüm.
2. **Sentetik görüntüler yalnızca eğitimde.** Doğrulama ve test setleri
   tamamen gerçek görüntülerden oluşuyor.
3. **GAN eğitiminde sınıf dengesi.** `nv` sınıfı verinin yaklaşık üçte ikisi.
   GAN her sınıfı eşit sıklıkta görsün diye `WeightedRandomSampler` kullanılıyor.
4. **Keras yerine PyTorch.** Tezde kullanılan GoogLeNet'in (Inception v1)
   ImageNet ağırlıklı hazır sürümü torchvision'da var, Keras'ta yok.
5. **Normalizasyon.** Tezde pikseller 0–1 aralığına getiriliyordu. Önceden
   eğitilmiş modeller ImageNet ortalama ve standart sapmasıyla eğitildiği için
   kodda bu normalizasyon kullanılıyor.

## Tez dokümanında fark edilen tutarsızlıklar

Kodda tek bir değer seçmek gerekti. Tezi bir yerde paylaşacaksan bunları
düzeltmen iyi olur:

- Görüntü boyutu bir yerde **60x60**, başka bir yerde **224x224**. Kodda
  sınıflandırıcılar için 224, GAN için 64 kullanıldı.
- Epoch sayısı bir yerde **50**, başka bir yerde **20**. Kodda 20 seçildi.
- Metot bölümünde **SqueezeNet** geçiyor ama hiçbir yerde kullanılmıyor.
- Sonuç tablosunda yalnızca klasik makine öğrenmesi modelleri var. CNN'lerin
  ve GAN'ın sonuçları raporlanmamış. Bu depo ikisini de raporluyor.
- [8]–[10] numaralı kaynaklar `example.com` yer tutucuları. Şekil numaraları
  tekrar ediyor (iki tane "Şekil 1.1"). Simgeler listesi şablondan kalma.

## Mülakatta anlatırken

- **Problem:** Cilt lezyonu verisi çok dengesiz. Nadir ama tehlikeli sınıflarda
  (ör. melanom) model zayıf kalıyor.
- **Çözüm:** Koşullu GAN ile nadir sınıflara sentetik görüntü üretip transfer
  öğrenme modellerini bu görüntülerle güçlendirmek.
- **Mühendislik:** Lezyon düzeyinde bölmeyle veri sızıntısını önleme, test
  setini tamamen gerçek tutma, tekrarlanabilir CLI, pytest ve CI.
- **Kısıtlar:** 64x64 GAN çıktısı bulanık; HAM10000 sınırlı bir popülasyondan
  geliyor; tıbbi teşhis aracı değil.
