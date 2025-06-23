# New ESRGAN ile Doğa Görüntüleri Netleştirme
[Türkçe](readme.md) | [English](readme_en.md)

Bu proje, derin öğrenme tabanlı **ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)** modeli kullanılarak düşük çözünürlüklü doğa görüntülerinin kalite artırımı üzerine gerçekleştirilmiştir.

## 📌 Proje Tanımı

Görüntü süper çözünürlük (Super-Resolution), düşük çözünürlüklü (LR) bir görüntüden yüksek çözünürlüklü (HR) bir görüntü üretmeyi hedefler. Bu projede ESRGAN mimarisi ile bulanıklaştırılmış ya da kalite düşürülmüş doğa manzaraları netleştirilmiştir.

---

## 🖼️ Örnek Çıktılar
Aşağıda modelin düşük çözünürlüklü girişlerden ürettiği yüksek çözünürlüklü çıktılara ait örnekler yer almaktadır:

###  Test Görüntüsü 1

![k1](https://github.com/user-attachments/assets/ac433516-1f27-4318-8490-f87b52d1dc8e)  

###  Test Görüntüsü 2

![k2](https://github.com/user-attachments/assets/cfc05f17-2b7c-4a35-a56a-6e223e39a7aa) 

###  Test Görüntüsü 3

![k3](https://github.com/user-attachments/assets/df9b2ac2-1060-4e80-b128-64471668377b) 

###  TensorBoard Çıktısı

![ssim_psnr_vgg](https://github.com/user-attachments/assets/5efafb75-9410-46e9-8b6e-434025762732)

---

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

- **Ortam:** Yerel GPU + Çevrimiçi GPU 
- **Epoch:** 25 
- **Batch Size:** 32
- **Optimizer:** Adam
- **Scheduler:** `gan_custom`, `lambdaLR`, `ReduceLROnPlateau` gibi farklı planlayıcılar test edilmiştir.
- **Model Kaydı:** `save_checkpoint()` ve `load_checkpoint()` fonksiyonları ile yapılmıştır.

## 🔬 Deneysel Sonuçlar

| Model | PSNR | SSIM |
|-------|------|------|
| En iyi sonuç (Custom + Yeni Dataset) | **43.03** | **0.9841** |

Tensorboard çıktıları, görsel karşılaştırmalar ve kayıpların detaylı analizi proje dosyalarında yer almaktadır.

## 🔧 Proje Yapısı

```

├── main.py
├── readme.md
├── src
│   ├── esrgan
│   │   ├── model.py
│   ├── test.py
│   ├── train.py
│   └── utils
│       ├── config.py
│       ├── data_loaders.py
│       ├── dataset_from_folder.py
│       ├── dataset.py
│       ├── loss.py
│       ├── scheduler_select.py
│       └── utils.py
└── test_images

````

## 🏁 Başlangıç

Projeyi çalıştırmak için:

```bash
python3 main.py train False
````

Test için:

```bash
python3 main.py test False
```

README dosyanıza çıktı görselleri için uygun bir bölüm başlığı aşağıdaki şekilde ekleyebilirsiniz:

## 👥 Katkıda Bulunanlar

* **Muhammet Halak** 
* **Abdullah Sina Korkmaz** 

## 📚 Kaynaklar

* [ESRGAN Paper (Wang et al.)](http://arxiv.org/abs/1809.00219)
* Kaggle: [LHQ 1024 Dataset](https://www.kaggle.com/datasets/dimensi0n/lhq-1024)

---
