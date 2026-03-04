# 🧠 Child Mind Institute — Problematic Internet Use
### Kaggle Competition | Rank: **488 / 3,599 Teams** | Top ~14% 🥈

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Overview

This repository contains my solution for the **Child Mind Institute — Problematic Internet Use** Kaggle competition, where the goal was to predict the **Severity Impairment Index (SII)** of children based on their physical activity data, fitness assessments, and demographic features.

The competition aimed to develop a predictive model that can help identify children at risk of problematic internet usage patterns — a critical problem in child mental health. The target variable `sii` is an **ordinal multi-class label** (0, 1, 2, 3), evaluated using **Quadratic Weighted Kappa (QWK)**.

> **Final Result:** Rank **488 out of 3,599 teams** — Top **~14% globally**

---

## 📂 Dataset Description

The dataset is multimodal, combining:

| Data Type | Description |
|-----------|-------------|
| **Tabular (CSV)** | Demographics, physical measurements, fitness scores, sleep disturbance, internet usage hours |
| **Time-Series (Parquet)** | Actigraphy sensor readings (accelerometer data) per patient |
| **Target** | `sii` — Severity Impairment Index (0 = None, 1 = Mild, 2 = Moderate, 3 = Severe) |

**Key feature groups in the tabular data:**
- `Basic_Demos` — Age, sex, enrollment season
- `Physical` — BMI, height, weight, blood pressure, heart rate
- `Fitness_Endurance` — Max stage, endurance time
- `FGC` — Fitness Gram Cadence scores across multiple zones
- `BIA` — Bioelectrical impedance analysis (body composition)
- `PAQ_A / PAQ_C` — Physical Activity Questionnaire scores
- `SDS` — Sleep Disturbance Scale
- `PreInt_EduHx` — Internet hours per day

---

## 🔧 Approach & Methodology

### 1. Data Preprocessing

- Loaded training CSV and merged with time-series actigraphy statistics (mean, std, min, max, etc. extracted via `describe()`)
- Used `ThreadPoolExecutor` for parallel loading of parquet files across patient IDs
- Handled missing values: categorical season columns filled with `'Missing'` and encoded as integers
- Applied consistent label encoding across train and test sets to prevent data leakage

```python
def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    return df.describe().values.reshape(-1), filename.split('=')[1]
```

### 2. Feature Engineering

- Extracted **statistical aggregates** (8 stats × N sensor columns) from raw actigraphy time-series as flat feature vectors
- Combined time-series statistics with tabular features: ~100+ features total
- Categorical encoding of seasonal metadata columns using unique-value mapping

### 3. Model: Single LightGBM Regressor

The core model is a **LightGBM Regressor** trained with Optuna-tuned hyperparameters:

```python
Params = {
    'learning_rate': 0.03884,
    'max_depth': 12,
    'num_leaves': 413,
    'min_data_in_leaf': 14,
    'feature_fraction': 0.7988,
    'bagging_fraction': 0.7602,
    'bagging_freq': 2,
    'lambda_l1': 4.735,
    'lambda_l2': 4.735e-06
}
```

**Training Strategy:**
- `StratifiedKFold` with **5 folds** (stratified on `sii` target)
- Seed: `42` for reproducibility
- `n_estimators = 200`

### 4. Threshold Optimisation (QWK Maximisation)

Since the model outputs **continuous regression values** and the metric is Quadratic Weighted Kappa on ordinal classes, the predictions were post-processed using **optimised thresholds**:

```python
def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))
```

Optimal thresholds were found using **Nelder-Mead minimisation** on the OOF predictions:

```python
KappaOptimizer = minimize(evaluate_predictions,
                          x0=[0.5, 1.5, 2.5],
                          args=(y, oof_non_rounded),
                          method='Nelder-Mead')
```

### 5. Evaluation Metric

**Quadratic Weighted Kappa (QWK)** measures agreement between predicted and actual ordinal categories, with heavier penalties for larger disagreements.

| Split | QWK Score |
|-------|-----------|
| Mean Train QWK | ~0.409 |
| Mean Validation QWK | ~0.471 |
| Optimised OOF QWK | Best achieved via threshold tuning |

---

## 🏗️ Project Structure

```
cmi-problematic-internet-use/
│
├── cmi-best-single-model.ipynb   # Main solution notebook
├── submission.csv                 # Final Kaggle submission
└── README.md                      # This file
```

---

## 📦 Dependencies

```bash
pip install numpy pandas scikit-learn lightgbm catboost xgboost optuna scipy polars colorama tqdm
```

| Library | Purpose |
|---------|---------|
| `lightgbm` | Primary gradient boosting model |
| `catboost`, `xgboost` | Ensemble candidates |
| `optuna` | Hyperparameter tuning |
| `scipy.optimize` | Nelder-Mead threshold optimisation |
| `polars` | Fast dataframe processing |
| `sklearn` | StratifiedKFold, metrics, cloning |

---

## 🚀 How to Run

1. Clone the repository and set up the Kaggle dataset:
```bash
kaggle competitions download -c child-mind-institute-problematic-internet-use
```

2. Place the data in `/kaggle/input/child-mind-institute-problematic-internet-use/`

3. Open and run the notebook:
```bash
jupyter notebook cmi-best-single-model.ipynb
```

4. The notebook will output `submission.csv` ready for Kaggle upload.

---

## 📊 Feature Importance

Top features by LightGBM gain importance included:
- `BIA-BIA_BMI`, `Physical-BMI`, `Physical-Weight` — Body composition metrics
- `SDS-SDS_Total_Raw` — Sleep disturbance scores
- `PreInt_EduHx-computerinternet_hoursday` — Internet usage hours
- `PAQ_A-PAQ_A_Total`, `PAQ_C-PAQ_C_Total` — Physical activity levels
- Time-series `Stat_*` features from actigraphy aggregates

---

## 🏆 Competition Result

| Metric | Value |
|--------|-------|
| Final Rank | **488 / 3,599** |
| Percentile | Top **~14%** |
| Evaluation Metric | Quadratic Weighted Kappa (QWK) |

---

## 📝 Key Learnings

- **Ordinal regression with threshold optimisation** outperforms direct classification when QWK is the metric
- **Time-series statistical aggregation** is a fast and effective way to incorporate sensor data into tabular models
- **Nelder-Mead threshold search** on OOF predictions is a reliable post-processing strategy for ordinal targets
- Stratified K-Fold is critical when the target class distribution is imbalanced

---



# 🦴 RSNA 2024 — Lumbar Spine Degenerative Classification
### Kaggle Competition | Medical Imaging | Deep Learning

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org/)
[![TIMM](https://img.shields.io/badge/TIMM-EdgeNeXt-orange)](https://github.com/rwightman/pytorch-image-models)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Overview

This repository contains my solution for the **RSNA 2024 Lumbar Spine Degenerative Classification** Kaggle competition, hosted by the Radiological Society of North America (RSNA). The objective was to automatically classify the **severity of five degenerative spinal conditions** across **five lumbar vertebral levels** from multi-series MRI scans.

The competition required working with real-world clinical DICOM imaging data, making it one of the most technically demanding biomedical machine learning challenges on Kaggle. The task directly addresses a critical need in radiology: reducing the burden on radiologists by automating the detection and grading of common lumbar spine conditions.

---

## 🏥 Clinical Context

Lumbar spine degeneration is among the most common causes of chronic back pain worldwide. This competition focused on classifying the following **5 conditions** at **5 spinal levels** (L1/L2 through L5/S1):

| Condition | Description |
|-----------|-------------|
| **Spinal Canal Stenosis** | Narrowing of the central spinal canal |
| **Left Neural Foraminal Narrowing** | Left-side nerve exit narrowing |
| **Right Neural Foraminal Narrowing** | Right-side nerve exit narrowing |
| **Left Subarticular Stenosis** | Left lateral recess narrowing |
| **Right Subarticular Stenosis** | Right lateral recess narrowing |

Each condition at each level is graded as: **Normal/Mild**, **Moderate**, or **Severe** — making this a **25-label, 3-class ordinal classification problem** (75 output nodes total).

---

## 📂 Dataset Description

| Data Type | Description |
|-----------|-------------|
| **DICOM Images** | Multi-series MRI scans per study (Sagittal T1, Sagittal T2/STIR, Axial T2) |
| **Series Descriptions** | CSV mapping study IDs to series types |
| **Labels** | 75 columns: `{condition}_{level}_{severity}` |
| **Format** | `.dcm` (DICOM) files, one folder per study/series |

**Three MRI series per patient study:**
- `Sagittal T1` — 14 slices selected per study
- `Sagittal T2/STIR` — 14 slices selected per study
- `Axial T2` — 14 slices selected per study
- **Total input tensor:** 42 channels × 512 × 512

---

## 🔧 Approach & Methodology

### 1. DICOM Image Processing

Raw DICOM files were decoded using `pydicom`, pixel arrays normalised to [0, 255], and resized to 512×512 using cubic interpolation:

```python
def read_dcm_image(self, src_path):
    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    norm_img = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    resized_img = cv2.resize(norm_img, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    return resized_img.astype(np.uint8)
```

### 2. Slice Sampling Strategy

Rather than using all slices (which vary in number per scan), a **uniform temporal sampling** strategy was applied to extract exactly 14 representative slices per series:

```python
step = len(img_paths) / 14.0
mid_point = len(img_paths) / 2.0 - 6.0 * step
for j, i in enumerate(np.arange(mid_point, len(img_paths), step)):
    idx = max(0, int(round(i - 0.5)))
    images[..., j] = self.read_dcm_image(img_paths[idx])
```

This ensures spatial coverage of the full spine regardless of the number of original slices.

### 3. Custom Dataset: `RSNA24TestDataset`

A PyTorch `Dataset` class was built to:
- Accept a DataFrame of study IDs and series descriptions
- Load all three MRI series (Sagittal T1, Sagittal T2/STIR, Axial T2) per study
- Stack them into a **42-channel input tensor** (14 channels × 3 series)
- Apply Albumentations transforms (resize + normalise)
- Output tensor shape: `[42, 512, 512]` — channels-first for PyTorch

```python
x[..., :14]   = self.load_series_images(study_id, 'Sagittal T1', 0)
x[..., 14:28] = self.load_series_images(study_id, 'Sagittal T2/STIR', 14)
x[..., 28:]   = self.load_series_images(study_id, 'Axial T2', 28)
```

### 4. Model Architecture: EdgeNeXt

The backbone used is **`edgenext_base.in21k_ft_in1k`** from the TIMM library — a lightweight yet powerful architecture combining edge convolutions with next-block designs, pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k.

```python
class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_c=42, n_classes=75, pretrained=True, features_only=False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=features_only,
            in_chans=in_c,       # 42-channel input
            num_classes=n_classes,  # 75 output nodes (25 labels × 3 classes)
            global_pool='avg'
        )
```

| Config | Value |
|--------|-------|
| Model | `edgenext_base.in21k_ft_in1k` |
| Input channels | 42 (14 slices × 3 series) |
| Output nodes | 75 (25 conditions/levels × 3 severity classes) |
| Image size | 512 × 512 |
| Folds | 5-Fold Cross-Validation |

### 5. Inference Pipeline

- Loaded **2 fold checkpoints** for ensemble averaging
- Used **mixed-precision inference** (`torch.cuda.amp.autocast` with `float16`)
- Per-study predictions averaged across all models
- Softmax applied per condition block (every 3 output nodes)

```python
for model in models:
    y = model(x)[0]
    for col in range(N_LABELS):
        pred = y[col * 3: (col + 1) * 3]
        y_pred = pred.float().softmax(dim=0).cpu().numpy()
        pred_per_study[col] += y_pred / len(models)
```

### 6. Submission Format

Output is a CSV with `row_id` formatted as `{study_id}_{condition}_{level}` and three probability columns for Normal/Mild, Moderate, and Severe grades:

```
row_id,normal_mild,moderate,severe
12345_spinal_canal_stenosis_l1_l2,0.85,0.10,0.05
...
```

---

## 🏗️ Project Structure

```
rsna-2024-lumbar-spine/
│
├── rsna-2024-lumbar-spine-prediction.ipynb  # Full inference notebook
├── submission.csv                            # Final Kaggle submission
└── README.md                                 # This file
```

---

## 📦 Dependencies

```bash
pip install torch torchvision timm albumentations pydicom opencv-python pandas numpy tqdm transformers
```

| Library | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `timm` | Pre-trained EdgeNeXt model |
| `albumentations` | Image augmentation and transforms |
| `pydicom` | Reading DICOM medical images |
| `cv2` (OpenCV) | Image resizing and processing |
| `transformers` | Cosine LR scheduler (training phase) |

---

## 🚀 How to Run

1. Download the competition data:
```bash
kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification
```

2. Place checkpoint files in `/kaggle/input/rsna-2024-edgenext-base/`:
   - `model_fold-1.pt`
   - `model_fold-2.pt`

3. Run the inference notebook:
```bash
jupyter notebook rsna-2024-lumbar-spine-prediction.ipynb
```

4. `submission.csv` will be generated in the working directory.

---

## ⚙️ Training Configuration (Reference)

| Parameter | Value |
|-----------|-------|
| Seed | 8620 |
| Image Size | 512 × 512 |
| Input Channels | 42 |
| Output Classes | 75 |
| Folds | 5 |
| Batch Size | 1 |
| Optimiser | AdamW |
| Scheduler | Cosine with warmup |
| Mixed Precision | FP16 (AMP) |
| EMA | ModelEmaV2 |

---

## 📝 Key Learnings

- **Multi-channel 2D CNN** is an effective strategy for volumetric MRI data — stacking slices as channels avoids the memory cost of full 3D convolutions
- **Uniform slice sampling** across varying-length scan sequences ensures consistent spatial coverage during both training and inference
- **DICOM preprocessing** requires careful normalisation: `(pixel - min) / (max - min + ε)` prevents division-by-zero artifacts
- **Mixed-precision inference** (`float16`) dramatically reduces memory consumption with negligible accuracy loss
- **Multi-fold ensemble averaging** at the probability level (before argmax) consistently outperforms single-model inference
