# 🌍 AQI Classification using Multimodal Deep Learning

**Course:** IT549 — Deep Learning Lab  
**Assignment:** Lab 3 — AQI Image Classification  
**Name:** [Your Name]  
**ID:** [Your Student ID]

---

## 📋 Objective

Classify **Air Quality Index (AQI)** into 6 categories using a **multimodal deep learning** approach that combines:

- **Visual data** — Images of locations (sky, haze, visibility)
- **Numerical data** — Pollutant sensor readings (`PM2.5`, `PM10`, `O3`, `CO`, `SO2`, `NO2`)

The project implements and compares two multimodal architectures:

1. **Multimodal CNN** — A custom CNN (image branch) + MLP (numerical branch), trained from scratch
2. **Multimodal ResNet18** — A pretrained ResNet18 backbone (first 10 layers frozen) + MLP, via transfer learning

---

## 📂 Project Structure

```
DL Lab_3/
├── Lab_3.ipynb                              # Main notebook (all code & results)
├── data (1).csv                             # CSV with filenames, AQI values & pollutant readings
├── sampled_images/                          # Folder containing 6,000 AQI images
├── Models/                                  # Saved model weights directory
├── best_basic_cnn.pth                       # Best weights — Multimodal CNN
├── best_pretrained_resnet18.pth             # Best weights — Multimodal ResNet18
├── IT549_Lab3_AQI_Image_Classification_Assignment.pdf  # Assignment PDF
└── README.md                                # This file
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Total samples** | 6,000 images |
| **Classes** | 6 — `Good`, `Moderate`, `Unhealthy for Sensitive Groups`, `Unhealthy`, `Very Unhealthy`, `Severe` |
| **Distribution** | Balanced — 1,000 images per class |
| **Image source** | Outdoor photos from multiple Indian cities (Bangalore, Tamil Nadu, Maharashtra) |
| **Split** | Train 70% (4,200) / Validation 15% (900) / Test 15% (900) |

### Numerical Features Used

| Feature | Description |
|---|---|
| `PM2.5` | Fine particulate matter (μg/m³) |
| `PM10` | Coarse particulate matter (μg/m³) |
| `O3` | Ozone concentration |
| `CO` | Carbon monoxide |
| `SO2` | Sulfur dioxide |
| `NO2` | Nitrogen dioxide |

> Missing values are imputed with the **median** (fitted on training set only) and all features are **standardized** using `StandardScaler`.

---

## 🧠 Model Architectures

### 1. Multimodal CNN (from scratch)

```
Image Branch:                   Numerical Branch:
┌─────────────────────┐        ┌─────────────────────┐
│ Conv2d(3→32) + BN   │        │ Linear(6→32) + ReLU │
│ Conv2d(32→64) + BN  │        │ BatchNorm1d(32)     │
│ Conv2d(64→128) + BN │        └─────────┬───────────┘
│ Conv2d(128→256) + BN│                  │
│ AdaptiveAvgPool(1×1)│                  │
└─────────┬───────────┘                  │
          │          ┌───────────┐       │
          └─────────►│ Concat    │◄──────┘
                     │ (256+32)  │
                     └─────┬─────┘
                           │
                     ┌─────▼─────┐
                     │ FC(288→128)│
                     │ FC(128→6)  │
                     └───────────┘
```

### 2. Multimodal ResNet18 (transfer learning)

- **Image branch:** Pretrained ResNet18 with first 10 layers **frozen** → 512-dim features
- **Numerical branch:** Linear(6→32) + ReLU + BatchNorm → 32-dim features
- **Fusion:** Concatenate (512+32=544) → FC(544→256) → FC(256→6)

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Epochs | 5 |
| Image Size | 224 × 224 |
| Augmentation | RandomHorizontalFlip, RandomRotation(15°), ColorJitter |
| Normalization | ImageNet mean/std |

---

## 📈 Tasks Completed

| Task | Description | Notebook Step |
|---|---|---|
| **Task 1** | Data preparation — load CSV, keep numerical columns, split, impute & scale | Step 2 |
| **Task 2** | Build Multimodal CNN from scratch (image + numerical branches) | Step 4 |
| **Task 3** | Transfer learning — Multimodal ResNet18 with frozen layers | Step 7 |
| **Task 4** | Evaluation — Classification report, confusion matrix, comparison table | Step 8 |
| **Task 5** | Training curves — Loss & accuracy plots for both models | Step 9 |
| **Task 6** | Misclassification analysis — Visualize errors & discuss reasons | Step 10 |

---

## 🚀 How to Run

### Prerequisites

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn Pillow
```

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nim369/Deep-Learning-Lab-3.git
   cd Deep-Learning-Lab-3
   ```

2. **Ensure `sampled_images/` folder** contains the 6,000 images and `data (1).csv` is present.

3. **Open and run the notebook:**
   ```bash
   jupyter notebook Lab_3.ipynb
   ```
   Run all cells sequentially from top to bottom (Kernel → Restart & Run All).

---

## 🔑 Key Insights

- **Multimodal fusion** (image + numerical features) provides richer context than images alone — pollutant readings help disambiguate visually similar conditions.
- **Adjacent AQI classes** (e.g., Moderate vs Unhealthy for Sensitive Groups) are the hardest to distinguish visually, as haze differences are subtle.
- **Transfer learning** with ResNet18 converges faster and generally achieves better accuracy than a CNN trained from scratch.
- **Weather effects** (fog, mist, clouds) can mimic pollution haze, leading to misclassifications.

---

## 📄 License

This project is for academic purposes as part of the IT549 Deep Learning Lab course.
