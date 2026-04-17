

---

## Files

```
aerial_project/
├── Aerial_Object_Classification.ipynb   ← Open in Google Colab (all training here)
├── app.py                               ← Streamlit app (run locally in VS Code)
├── requirements.txt                     ← pip install -r requirements.txt
└── models/                              ← Put trained model files here after Colab
    ├── best_model.keras
    └── yolov8_best.pt   (optional)
```

---

## Step-by-step Workflow

### Step 1 — Train in Google Colab

1. Upload `Aerial_Object_Classification.ipynb` to Colab
2. Set Runtime → **GPU → T4**
3. Upload both datasets to Google Drive:
   ```
   MyDrive/aerial_project/classification_dataset/   ← TRAIN / VALID / TEST inside
   MyDrive/aerial_project/object_detection_Dataset/ ← train / valid / test inside
   ```
4. Update `BASE_DRIVE` in **Section 1** to your Drive path
5. Run all cells top to bottom
6. Trained models auto-save to `MyDrive/aerial_project/saved_models/`

### Step 2 — Download Models

After Colab finishes:
- Download `best_model.keras` from Drive → `saved_models/`
- Download `yolov8_best.pt` from Drive → `saved_models/` (if you ran Section 6)

### Step 3 — Run the Streamlit App in VS Code

```bash
# Install dependencies
pip install -r requirements.txt

# Create models folder and drop your downloaded files in
mkdir models
# copy best_model.keras and yolov8_best.pt into models/

# Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Dataset Structure Expected

```
classification_dataset/
├── TRAIN/
│   ├── bird/     (1414 images)
│   └── drone/    (1248 images)
├── VALID/
│   ├── bird/     (217 images)
│   └── drone/    (225 images)
└── TEST/
    ├── bird/     (121 images)
    └── drone/    (94 images)

object_detection_Dataset/
├── train/
│   ├── images/
│   └── labels/   (.txt files in YOLO format)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## What the Colab Notebook Covers

| Section | Content |
|---------|---------|
| 1 | Drive mount, path config, imports |
| 2 | EDA: class counts, sample images, dimensions, pixel distributions, augmentation preview |
| 3 | Custom CNN: architecture, training, evaluation, confusion matrix |
| 4 | Transfer Learning: ResNet50 → MobileNetV2 → EfficientNetB0 (2-phase each) |
| 5 | Model comparison: table, bar charts, radar chart, best model selection |
| 6 | YOLOv8: data.yaml, annotation preview, training, validation, inference |

---

## Tech Stack

- **TensorFlow / Keras** — classification
- **Ultralytics YOLOv8** — object detection
- **Streamlit** — web app
- **scikit-learn** — metrics
- **Matplotlib / Seaborn** — plots
