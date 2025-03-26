# Dual-Stream Convolutional Neural Network with Self-Attention on UCI HAR Dataset

### Done as part of Major Project on Comparative Analysis of Hybrid Anomaly Detection Approaches in Fall Detection

## Overview
The primary objective was to detect falls and classify human activities using a Dual-Stream Convolutional Neural Network with Self-Attention (CNN-SA) architecture applied to the UCI Human Activity Recognition Dataset.

The project was conducted as part of a major academic initiative to explore advanced methods for fall detection, which is critical in healthcare and smart-home applications.

---

## Features
- **Dual-Stream CNN Architecture**:
  - Extracts both local temporal features and global contextual features.
- **Self-Attention Mechanism**:
  - Enhances the model's ability to focus on important activity patterns.
- **Class Imbalance Handling**:
  - Applied class weighting and oversampling techniques to improve classification accuracy.
- **Performance Metrics**:
  - Evaluated model performance using precision, recall, F1-score, and confusion matrix.

---

## Dataset
### UCI Human Activity Recognition Using Smartphones Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- **Description**:
  - Contains sensor data from smartphones (accelerometer and gyroscope).
  - Six activity classes: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying.

---

## Methodology
### Architecture
1. **Dual-Stream CNN**:
   - Stream 1: Extracts short-term temporal features using 3×1 convolution filters.
   - Stream 2: Extracts long-term contextual features using 5×1 convolution filters.
2. **Self-Attention Mechanism**:
   - Multi-head attention layer focuses on relevant temporal patterns.
3. **Classifier**:
   - Fully connected layers with dropout and L2 regularization for robust classification.

### Steps
1. Data preprocessing:
   - Normalized the dataset using z-score normalization.
   - Reshaped data into `(samples, timesteps, features)` format for CNN input.
2. Model training:
   - Used cyclical learning rate scheduling for stable convergence.
   - Applied class weighting to address imbalance in activity classes.
3. Evaluation:
   - Generated confusion matrix and classification report for detailed analysis.

---

## Results
### Confusion Matrix - edit
![Confusion Matrix](./file/image_cf.png)

### Key Metrics
| Metric       | Value       |
|--------------|-------------|
| **Test Accuracy** | 85.44%      |
| **Precision**     | 86.07%      |
| **Recall**        | 84.90%      |

### Classification Report
| Class                | Precision | Recall | F1-Score |
|----------------------|-----------|--------|----------|
| Walking              | 0.87      | 0.89   | 0.88     |
| Walking Upstairs     | 0.82      | 0.87   | 0.85     |
| Walking Downstairs   | 0.84      | 0.77   | 0.80     |
| Sitting              | 0.79      | 0.79   | 0.79     |
| Standing             | 0.85      | 0.79   | 0.82     |
| Laying               | 0.94      | 0.99   | 0.97     |

---

## Installation and Usage

### Requirements - edit
Install the following Python packages:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn imbalanced-learn
```

### Steps to Run the Project - edit cd
1. Clone the repository:
    ```bash
    git clone https://github.com/nsdv-frctrd/CNN-SA_UCI-HAR.git
    cd gyro_acce
    ```
2. Upload the UCI HAR dataset to your working directory or Google Drive.
3. Run the Colab notebook `gyro_acce.ipynb`:
    - Train the model or load the saved model (`best_model.keras`) for evaluation.
4. Visualize outputs such as confusion matrix and training history.

---

## File Structure - edit 
```
fall-detection-hybrid-anomaly/
├── README.md                # Project documentation
├── fall_detection.ipynb     # Colab notebook with code implementation
├── best_model.keras         # Saved trained model (Keras format)
├── image_cf.jpg                # Confusion matrix visualization
├── training_history.csv     # Training metrics (accuracy, loss)
└── UCI_HAR_Dataset/         # Dataset folder (optional)
```

---

## References
1. UCI Machine Learning Repository: [Human Activity Recognition Using Smartphones Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
2. TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)

---

## License
This project is licensed under the MIT License.

---
