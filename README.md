# Plant Disease Detection Using Deep Learning and Explainable AI

A complete starter project for classifying plant-leaf images and explaining predictions with Grad-CAM. It uses TensorFlow/Keras, MobileNetV2 transfer learning, Streamlit, and optional LIME explanations.

> **Important:** This is an educational decision-support prototype, not a substitute for diagnosis by an agricultural expert. Dataset performance may not transfer to field photographs. Do not apply pesticides solely on the basis of this model.

## Features
- Train MobileNetV2 on a folder-structured plant disease dataset
- Train/validation/test split with class names saved alongside the model
- Evaluation report, confusion matrix, and classification report
- Streamlit image-upload application
- Predicted class and confidence scores
- Grad-CAM heatmap overlay to visualize influential image regions
- Optional LIME superpixel explanation
- Configurable image size, batch size, and epochs

## 1. Requirements
- Python 3.10 or 3.11 recommended
- 8 GB RAM recommended for training
- Internet access for the first run if pretrained ImageNet weights are not cached
- GPU optional

## 2. Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Prepare the dataset
Download PlantVillage (or another compatible leaf-image dataset). Organize it so each class has its own folder:

```text
data/PlantVillage/
  Tomato___Early_blight/
    image1.jpg
    image2.jpg
  Tomato___healthy/
    image3.jpg
  Potato___Late_blight/
    image4.jpg
```

The folder names become the class labels. The training script recursively reads supported image files inside each class folder.

Dataset sources:
- PlantVillage on Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease
- PlantDoc (more field-like images; requires compatible class-folder organization): https://github.com/pratikkayal/PlantDoc-Dataset

Place the extracted dataset at `data/PlantVillage/`, or pass another path with `--data_dir`.

## 4. Train
From the project root:
```bash
python train.py --data_dir data/PlantVillage --epochs 10 --batch_size 32
```

For a quick smoke test, use a small subset copied into class folders and `--epochs 1`.

The script creates:
- `models/plant_disease_mobilenetv2.keras`
- `models/class_names.json`
- `outputs/classification_report.txt`
- `outputs/confusion_matrix.png`
- `outputs/training_history.png`

If training data is limited, use fewer classes first. Keep the test set separate and avoid near-duplicate images across splits when possible.

## 5. Run the web app
After training:
```bash
streamlit run app.py
```
Upload a JPG, JPEG, or PNG leaf image. The app displays the predicted class, confidence, class probabilities, and Grad-CAM overlay.

## 6. Optional LIME explanation
Install the optional dependency:
```bash
pip install lime scikit-image
```
Then check “Generate LIME explanation” in the app. LIME is slower than Grad-CAM and is intended as an additional local explanation.

## 7. Project structure
```text
plant_disease_xai_project/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── model_utils.py
│   ├── explainability.py
│   └── evaluation.py
├── data/                 # Put downloaded dataset here
├── models/               # Trained model and class names are saved here
└── outputs/              # Evaluation outputs
```

## 8. Model and explainability notes
- MobileNetV2 is initialized with ImageNet weights and fine-tuned for the dataset classes.
- Input images are resized to 224×224 and preprocessed with the MobileNetV2 preprocessing function.
- Grad-CAM uses gradients of the predicted class score with respect to the final convolutional feature map.
- A Grad-CAM heatmap highlights regions influential to the prediction; it does **not** prove that the highlighted region is a biological cause of disease.
- Confidence is a model score, not a calibrated probability of clinical/agronomic certainty.

## 9. Recommended experiments for a report
1. Compare a small custom CNN with MobileNetV2.
2. Compare frozen-feature training with fine-tuning.
3. Report accuracy, macro precision, macro recall, macro F1, and confusion matrix.
4. Inspect Grad-CAM examples for correct and incorrect predictions.
5. Test on images from a different source (e.g., PlantDoc) to measure domain shift.
6. Discuss class imbalance, background bias, lighting, camera differences, and dataset leakage.

## Troubleshooting
- **No model found:** train first; the app expects `models/plant_disease_mobilenetv2.keras`.
- **Dataset not found:** ensure each class is a subfolder of the data directory.
- **Out of memory:** reduce `--batch_size` to 8 or 16.
- **Poor field performance:** PlantVillage images often have controlled backgrounds. Add diverse field images, validate externally, and avoid claiming deployment readiness based only on PlantVillage test accuracy.
