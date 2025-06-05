Bu parametreler DataLoader'ın performansını önemli ölçüde artıran optimizasyon özellikleridir:

## `persistent_workers=True`

**Ne yapar:**
- Normal durumda her epoch sonunda worker process'ler kapatılır ve yeni epoch için yeniden başlatılır
- `persistent_workers=True` ile worker'lar epoch'lar arası canlı kalır

**Faydaları:**
```python
# Normal durum (persistent_workers=False):
# Epoch 1: Worker'lar başlatılır → Veri yüklenir → Worker'lar kapatılır
# Epoch 2: Worker'lar yeniden başlatılır → Veri yüklenir → Worker'lar kapatılır
# ...

# persistent_workers=True ile:
# Epoch 1: Worker'lar başlatılır → Veri yüklenir → Worker'lar açık kalır
# Epoch 2: Aynı worker'lar devam eder → Veri yüklenir → Worker'lar açık kalır
# ...
```

**Performans artışı:**
- Worker başlatma/kapatma maliyeti ortadan kalkar
- Özellikle çok epoch'lu eğitimlerde büyük zaman tasarrufu
- Memory'de önceden yüklenmiş veriler korunabilir

## `prefetch_factor=2`

**Ne yapar:**
- Her worker'ın bellekte hazır tutacağı batch sayısını belirler
- Default değer genellikle 2'dir, ama explicitly belirtmek iyi practice'tir

**Nasıl çalışır:**
```python
# prefetch_factor=2 ile:
# Worker 1: Batch 1 işleniyor, Batch 3 hazırlanıyor
# Worker 2: Batch 2 işleniyor, Batch 4 hazırlanıyor
# Model: Batch 0'ı eğitimde kullanıyor

# Bu sayede GPU hiç beklemez, sürekli veri akışı olur
```

**Faydaları:**
- GPU beklemesini minimize eder
- CPU ve GPU arasında pipeline oluşturur
- Overall throughput artışı

## Pratik Örnek:

````python
# Optimized DataLoader configuration
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=4,                # CPU core sayısına göre ayarlayın
    pin_memory=True,             # GPU transfer'i hızlandırır
    persistent_workers=True,     # Worker'ları epoch arası canlı tut
    prefetch_factor=2,          # Her worker 2 batch önceden hazırlasın
    drop_last=True,             # Son incomplete batch'i atla (training için)
)
````

## Dikkat Edilecek Noktalar:

**Memory kullanımı:**
```python
# Total memory usage:
# num_workers × prefetch_factor × batch_size × data_size_per_sample
# Örnek: 4 workers × 2 prefetch × 32 batch × image_size = Significant RAM usage
```

**Optimal ayarlar:**
- `num_workers`: CPU core sayısının 1/2 - 3/4'ü
- `prefetch_factor`: 1-4 arası (çok yüksek değerler RAM tüketir)
- `persistent_workers`: Çoğu durumda True yapın

Bu optimizasyonlar özellikle büyük veri setleri ve uzun eğitimler için %20-40 performans artışı sağlayabilir.