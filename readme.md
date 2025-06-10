# ESRGAN ile Doğa Görüntüleri Netleştirme

Bu proje, derin öğrenme tabanlı **ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)** modeli kullanılarak düşük çözünürlüklü doğa görüntülerinin kalite artırımı üzerine gerçekleştirilmiştir.

## 📌 Proje Tanımı

Görüntü süper çözünürlük (Super-Resolution), düşük çözünürlüklü (LR) bir görüntüden yüksek çözünürlüklü (HR) bir görüntü üretmeyi hedefler. Bu projede ESRGAN mimarisi ile bulanıklaştırılmış ya da kalite düşürülmüş doğa manzaraları netleştirilmiştir.

## 🧠 Kullanılan Model: ESRGAN

- **Generator:** Residual-in-Residual Dense Block (RRDB) mimarisi ile detayların korunması ve geliştirilmesi.
- **Discriminator:** Relativistic GAN yaklaşımı ile daha gerçekçi görüntüler üretme.
- **Kayıp Fonksiyonları:**
  - VGG Perceptual Loss
  - Adversarial Loss
  - L1 Loss

## 📁 Veri Kümesi

- **Dataset:** [Landscapes dataset (LHQ 1024×1024)](https://www.kaggle.com/datasets/dimensi0n/lhq-1024)
- **İşlem Adımları:**
  - Yeniden boyutlandırma
  - HR-LR çiftlerinin oluşturulması
  - Normalizasyon
  - Kırpma (HR: 128x128, LR: 32x32)

## ⚙️ Eğitim Detayları

- **Ortam:** Google Colab GPU + Yerel GPU
- **Epoch:** 25 (ön denemeler dahil)
- **Batch Size:** 16
- **Optimizer:** Adam
- **Scheduler:** `gan_custom`, `lambdaLR`, `ReduceLROnPlateau` gibi farklı planlayıcılar test edilmiştir.
- **Model Kaydı:** `save_checkpoint()` ve `load_checkpoint()` fonksiyonları ile yapılmıştır.

## 🔬 Deneysel Sonuçlar

| Model | PSNR | SSIM |
|-------|------|------|
| En iyi sonuç (Custom + Yeni Dataset) | **30.99** | **0.693** |

Tensorboard çıktıları, görsel karşılaştırmalar ve kayıpların detaylı analizi proje dosyalarında yer almaktadır.

## 🔧 Proje Yapısı

```

├── data/                   # Veri seti klasörü
├── models/                 # ESRGAN model bileşenleri
│   ├── generator.py
│   ├── discriminator.py
│   └── blocks.py
├── utils/
│   └── checkpoint.py       # Model kayıt/yükleme
├── train.py                # Eğitim scripti
├── test.py                 # Görüntü test scripti
├── config.py               # Parametreler
├── README.md

````

## 🏁 Başlangıç

Projeyi çalıştırmak için:

```bash
pip install -r requirements.txt
python train.py
````

Test için:

```bash
python test.py --image "test_images/sample.jpg"
```

## 👥 Katkıda Bulunanlar

* **Muhammet Halak** 
* **Abdullah Sina Korkmaz** 

## 📚 Kaynaklar

* [ESRGAN Paper (Wang et al.)](http://arxiv.org/abs/1809.00219)
* Kaggle: [LHQ 1024 Dataset](https://www.kaggle.com/datasets/dimensi0n/lhq-1024)

---
