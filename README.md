# Uncertainty Quantification in Medical XAI

**CENG 463 – Introduction to Machine Learning | Term Project**
*İzmir Institute of Technology (İYTE) — Spring 2026*

---

## Proje Hakkında

Bu proje, tıbbi görüntüleme (NIH Chest X-Ray14) üzerinde **belirsizlik tahminli** derin öğrenme modelleri geliştirir ve **Grad-CAM** görselleştirmeleri ile açıklanabilirlik sağlar.

| Model | Tür |
|---|---|
| Deterministic CNN (ResNet-50) | Baseline 1 |
| MC Dropout | Baseline 2 |
| **Evidential Deep Learning** | **Proposed** |

---

## Dosya Yapısı

```
300201040_463_FinalProject/
│
├── config.py        ← Tüm ayarlar: path, epoch, lr, batch size...
├── dataset.py       ← Veri yükleme, train/val/test bölme, transforms
├── models.py        ← Tüm modeller: DeterministicCNN, MCDropoutCNN
├── uncertainty.py   ← Metrikler: ECE, entropy, kalibrasyon grafiği
├── train.py         ← Model eğitimi (komut satırından çalışır)
├── evaluate.py      ← Test seti değerlendirmesi
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── raw/         ← Dataset buraya gelecek (git'e dahil değil)
│
└── results/
    ├── checkpoints/ ← Kaydedilen model ağırlıkları (.pth)
    ├── figures/     ← Kalibrasyon grafikleri, görselleştirmeler
    └── tables/      ← Metrik tabloları (.csv)
```

---

## Kurulum

### 1. Sanal ortam oluştur
```bash
python -m venv venv
venv\Scripts\activate      # Windows
```

### 2. Bağımlılıkları yükle
```bash
pip install -r requirements.txt
```

### 3. Dataset'i indir (NIH Chest X-Ray14)

Kaggle üzerinden:
```bash
kaggle datasets download -d nih-chest-xrays/data
```
Ardından dosyaları şu yapıya çıkar:
```
data/raw/
├── Data_Entry_2017.csv
└── images/
    ├── 00000001_000.png
    └── ...
```

---

## Kullanım

### Model Eğitimi

```bash
# Baseline 1: Deterministic CNN
python train.py --model resnet50

# Baseline 2: MC Dropout
python train.py --model mc_resnet50 --dropout_p 0.25

# Farklı backbone ile
python train.py --model resnet18 --epochs 15
```

### Model Değerlendirme

```bash
# Deterministic
python evaluate.py --model deterministic \
    --checkpoint results/checkpoints/deterministic_resnet50_best.pth

# MC Dropout (30 stokastik pass)
python evaluate.py --model mc_dropout \
    --checkpoint results/checkpoints/mc_dropout_resnet50_best.pth \
    --mc_samples 30
```

Çıktılar:
- `results/figures/<model>_reliability.png` — Kalibrasyon grafiği
- `results/tables/<model>_metrics.csv`      — Metrik tablosu

---

## Ayarlar (`config.py`)

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `BACKBONE` | `resnet50` | resnet18 / resnet50 / densenet121 |
| `NUM_EPOCHS` | `15` | Eğitim epoch sayısı |
| `BATCH_SIZE` | `32` | Batch büyüklüğü |
| `LR` | `1e-4` | Öğrenme hızı |
| `DROPOUT_P` | `0.25` | MC Dropout olasılığı |
| `MC_SAMPLES` | `30` | Stokastik forward pass sayısı |
| `PATIENCE` | `7` | Early stopping sabrı |

---

## GPU ve Veriseti Slicing (YENİ)

Proje, **NVIDIA RTX 4060** GPU'sunu kullanacak şekilde optimize edilmiştir. Ayrıca, tüm 40GB'lık veriyi işlemek yerine, performansı artırmak için 6GB'lık bir alt küme (subset) oluşturma imkanı eklenmiştir.

### 6GB Alt Küme Hazırlama

Veriseti `data/raw/images` altına yüklendikten sonra şu komutu çalıştırın:
```bash
python scratch/prepare_subset.py
```
Bu script, görselleri tarar ve toplamda tam 6GB olacak şekilde bir subset seçerek `Data_Entry_2017.csv` dosyasını günceller.

### GPU Kullanımı

`config.py` içinde `DEVICE = "cuda"` olarak ayarlanmıştır. Eğitim ve değerlendirme otomatik olarak GPU üzerinde çalışacaktır.

---

## Referanslar

- Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation.* ICML.
- Sensoy et al. (2018). *Evidential Deep Learning to Quantify Classification Uncertainty.* NeurIPS.
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV.
- Wang et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database.* CVPR.
 
 
 
 
 

<!-- version 18 -->

<!-- simulation-step-18 -->

<!-- contribution-fix-12 -->
