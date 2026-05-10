# Chest X-Ray Classification using Transfer Learning

**Health Informatics Research Lab (HIRL) — Computer Vision Project**

A deep learning system that classifies chest X-ray images into three categories:
**Normal**, **Bacterial Pneumonia**, and **Viral Pneumonia** — using Transfer Learning
with VGG16, ResNet50, and MobileNetV2.

---

## What This Project Does

This project simulates a real-world AI healthcare application. It takes a chest X-ray
image as input and predicts whether the patient's lungs appear normal or show signs of
bacterial or viral pneumonia. It also includes Grad-CAM heatmaps that visually show
*where* the model is looking when it makes a decision.

---

## Project Structure

```
chest-xray-classification/
│
├── Transfer_Learning_ChestXray.ipynb   ← Main training notebook
│
├── chest_xray/                         ← Original Kaggle dataset (download separately)
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── test/
│
│
├── chest_xray_3class/                  ← Auto-created by the notebook
│   ├── train/
│   │   ├── NORMAL/
│   │   ├── BACTERIA/
│   │   └── VIRUS/
│   ├── test/
│   
│
├── transfer_VGG16_xray.h5              ← Saved after training
├── transfer_ResNet50_xray.h5
├── transfer_MobileNetV2_xray.h5
│
├── class_distribution.png             ← Generated outputs
├── sample_images.png
├── augmented_samples.png
├── training_history_all_models.png
├── confusion_matrices_all.png
├── model_comparison.png
├── gradcam_all_models.png
└── gradcam_per_class_<model>.png
```

---

## Dataset

**Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Property | Details |
|---|---|
| Total images | 5,856 |
| Classes | Normal, Bacterial Pneumonia, Viral Pneumonia |
| Patient age group | Pediatric (1–5 years) |
| Split | Pre-split into train / test / val |
| Known issue | Class imbalance — pneumonia cases outnumber normal cases |

> The Kaggle dataset comes with only 2 folders (NORMAL and PNEUMONIA).
> The notebook automatically reorganizes it into 3 folders by reading the filenames,
> which contain either `bacteria` or `virus`.

---

## Models Used

All three models were pre-trained on ImageNet (1.2 million images, 1000 classes).
We replace their original output head with our own 3-class classifier.

| Model | Key Idea | Size |
|---|---|---|
| VGG16 | Classic stacked conv layers, simple and reliable | Large |
| ResNet50 | Skip connections prevent vanishing gradients, very deep | Medium |
| MobileNetV2 | Lightweight depthwise convolutions, fast inference | Small |

### Two-Phase Training Strategy

**Phase 1 — Freeze base, train top layers only**
The pre-trained base is frozen. Only our new classification layers learn.
Learning rate: `1e-3`. Fast and safe.

**Phase 2 — Fine-tune last 20 base layers**
The last 20 layers of the base model are unfrozen and fine-tuned
alongside our top layers using a very small learning rate (`1e-5`).
This lets the model adapt its deeper features to X-ray images.

---

## Requirements

```
tensorflow >= 2.10
numpy
matplotlib
seaborn
scikit-learn
opencv-python-headless
Pillow
```

Install all at once:

```bash
pip install tensorflow matplotlib seaborn scikit-learn numpy opencv-python-headless Pillow
```

---

## How to Run

### Option A — Google Colab (recommended, free GPU)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Transfer_Learning_ChestXray.ipynb`
3. Enable GPU: **Runtime → Change runtime type → T4 GPU**
4. Download the dataset from Kaggle and upload to your Google Drive
5. Update `ORIGINAL_DIR` in the notebook to your Drive path:
   ```python
   ORIGINAL_DIR = '/content/drive/MyDrive/chest_xray'
   ```
6. Run all cells top to bottom

### Option B — Local Machine

1. Clone or download this repository
2. Download the dataset from Kaggle and place it in the same folder as the notebook
3. Make sure `ORIGINAL_DIR = './chest_xray'` in the notebook
4. Run:
   ```bash
   jupyter notebook Transfer_Learning_ChestXray.ipynb
   ```

### Running Grad-CAM (after training)

After training is complete and models are saved as `.h5` files:


This notebook is fully self-contained — it reloads the saved models automatically.
No need to re-run training.

---

## Notebook Walkthrough

### Transfer_Learning_ChestXray.ipynb

| Step | What it does | Marks |
|---|---|---|
| 1 — Data Loading | Load images, count per class, visualize distribution, check dimensions | 5 |
| 2 — Preprocessing | Resize to 224×224, normalize, augment, split into train/val/test | 7 |
| 3 — Build Models | VGG16, ResNet50, MobileNetV2 with custom classification head | 7 |
| 4 — Train | Two-phase training with class weights, early stopping, LR reduction | — |
| 5 — Training Curves | Accuracy and loss plots with Phase 1 / Phase 2 boundary marked | 5 |
| 6 — Evaluate | Accuracy, Precision, Recall, F1-Score, Confusion Matrix | 6 |
| 7 — Predictions | Sample correct and incorrect predictions visualized | 5 |

### GradCAM_Fixed.ipynb

Loads saved models and generates Grad-CAM heatmaps showing where each model
focuses its attention when classifying an X-ray.

---

## Key Design Decisions

**Why Transfer Learning instead of CNN from scratch?**
The pre-trained models already know how to detect edges, textures, shapes, and
complex visual patterns from 1.2 million images. We only need to teach them the
final classification step. This gives higher accuracy with less data and less
training time.

**Why reorganize into 3 folders?**
The Kaggle dataset stores all pneumonia images (both types) in one `PNEUMONIA/` folder.
TensorFlow's `flow_from_directory` uses folder names as class labels, so with 2 folders
it creates a 2-class model. We reorganize into `NORMAL/`, `BACTERIA/`, and `VIRUS/`
so it correctly creates a 3-class model.

**Why class weights?**
The dataset has significantly more pneumonia images than normal images.
Without correction the model learns to just predict pneumonia for everything.
Class weights penalize mistakes on underrepresented classes more heavily.

**Why two-phase training?**
Jumping straight to fine-tuning with a large learning rate destroys the knowledge
the base model gained from ImageNet. Phase 1 first trains our new top layers to a
reasonable state, then Phase 2 makes small precise adjustments to the base layers.

---

## Understanding the Results

**Why is Bacteria vs Virus the hardest pair?**
Both types of pneumonia cause similar patterns in X-rays — cloudy or hazy areas
in the lungs. Even trained radiologists sometimes struggle to distinguish them
without additional clinical information. This is expected behavior.


**Model limitations**

- This is a screening tool, not a diagnostic replacement for doctors
- Trained only on pediatric patients aged 1–5
- Clinical deployment requires regulatory approval and extensive external validation
- Performance may differ on adult patients or images from different scanners

---

## File Descriptions

| File | Description |
|---|---|
| `Transfer_Learning_ChestXray.ipynb` | Main notebook — data, training, evaluation |
| `GradCAM_Fixed.ipynb` | Grad-CAM heatmaps — fully self-contained |
| `transfer_<model>_xray.h5` | Saved trained model weights |
| `class_distribution.png` | Bar chart of image counts per class |
| `sample_images.png` | Sample X-rays from each class |
| `augmented_samples.png` | Examples of augmented training images |
| `training_history_all_models.png` | Accuracy and loss curves for all 3 models |
| `confusion_matrices_all.png` | Confusion matrix for each model |
| `model_comparison.png` | Side-by-side metric comparison bar chart |
---

## Acknowledgements

- Dataset: [Paul Mooney on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Original dataset source: Kermany et al., *Cell*, 2018
- Project assigned by: Health Informatics Research Lab (HIRL), Daffodil International University

---

*Computer Vision Project | HIRL | Chest X-Ray Classification using Transfer Learning*
