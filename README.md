# Facial Emotion Recognition (FER2013)

End-to-end facial emotion classification across 7 emotions (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) using TensorFlow/Keras and `MobileNetV2` transfer learning.

## Project Files

- `facial_emotion_recognition_fer2013.ipynb` — Complete training, evaluation, and inference notebook.
- `facial_emotion_recognition.pdf` — Project report/notebook export.

## Pipeline Overview

1. **Dataset**: Downloads `msambare/fer2013` via `kagglehub` (`48x48` grayscale images normalized to `[0, 1]`).
   - Train / Validation: `28,709` images split 85/15 stratified (`24,402` train / `4,307` val).
   - Test: `7,178` images.
2. **Model (`FER2013_MobileNetV2_Transfer`)**:
   - In-model augmentation (`RandomFlip`, `RandomRotation`, `RandomTranslation`, `RandomZoom`)
   - 1-channel to 3-channel conversion + resize to `96x96`
   - Fine-tuned `MobileNetV2` (`imagenet` weights, `include_top=False`)
   - `GlobalAveragePooling2D` → `Dense(256, elu)` → `Dropout(0.4)` → `Dense(7, softmax)`
3. **Training**:
   - `Adam(lr=1e-3)`, `sparse_categorical_crossentropy`, balanced `class_weight`, `batch_size=64`, max `35` epochs
   - Callbacks: `ModelCheckpoint`, `ReduceLROnPlateau`, `EarlyStopping`
4. **Saved Artifacts & Inference**:
   - Saves `/content/models/fer2013_emotion_model.keras` and `/content/models/class_indices.json`
   - Includes `predict_emotion()` for single-image inference

## Test Results

| Metric | Value |
| :--- | :---: |
| Test Loss | 1.0811 |
| Test Accuracy | 59.71% |
| Macro Precision / Weighted Precision | 57.02% / 61.83% |
| Macro Recall / Weighted Recall | 57.15% / 59.71% |
| Macro F1-Score / Weighted F1-Score | 56.40% / 60.14% |

## Usage

Open `facial_emotion_recognition_fer2013.ipynb` in Google Colab (paths default to `/content/data/fer2013` and `/content/models`) or locally with:

```bash
pip install tensorflow numpy pandas matplotlib seaborn pillow scikit-learn kagglehub
```

Run all cells sequentially to download the dataset, train, evaluate, export the `.keras` model, and run sample inference.
