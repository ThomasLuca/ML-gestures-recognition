# Project Overview: Hand Gesture Recognition in Video

This project, developed for a university machine learning course by **Thomas, Felix, and Cobe**, focuses on classifying dynamic hand gestures from video data. Moving beyond static image analysis, the team explored various **motion extraction techniques** and deep learning architectures, specifically **CNNs (Convolutional Neural Networks)** and **RNNs (Recurrent Neural Networks)**, to identify four specific gestures: **Next, Rotateright, Scrolldown, and Zoomin**.

---

## Data Pipeline

The project implements a robust pipeline to handle raw video data and prepare it for model training.

### 1. Dataset Preparation & EDA

* **Restructuring**: The raw dataset was reorganized into a streamlined format (root > gesture > video > frames) to ensure consistency.
* **Exploratory Data Analysis (EDA)**: The team analyzed the distribution of videos per gesture and frames per video to identify anomalies. Unbalanced classes were addressed by limiting the number of videos per gesture to match the smallest class during training data preparation.
* **Outlier Handling**: Extremely short (e.g., single-frame) or unusually long videos were identified and filtered out to ensure data quality.



### 2. Preprocessing & Motion Extraction

The core challenge was transitioning from static images to capturing evolving movements. The team tested four motion extraction techniques:

* **Optical Flow (Farneback & Lucas Kanade)**: Tracked pixel shifts between frames but proved sensitive to specific video conditions.
* **Median Frame Differencing**: Effective at isolating movement but potentially too aggressive in discarding information.
* **Frame Differencing**: Subtraction of consecutive frames. This was selected for the final pipeline as it yielded superior results and was computationally efficient.

**Final Preprocessing Steps:**

1. **Resizing**: Frames reduced to $100\times100$ resolution.
2. **Movement Extraction**: Standard frame differencing.
3. **Length Equalization**: Videos upsampled or downsampled to a fixed length of **35 frames**.


---

## Model Architectures

The team experimented with several deep learning approaches to find the most effective classifier.

### Convolutional Neural Networks (CNN)

A **2D+1D Convolutional model** was implemented, inspired by standard video classification tutorials.

* **Structure**: Features four main layers utilizing **Conv3D** (decomposed into spatial and temporal operations), Layer Normalization, and ReLU activations.
* **Performance**: Achieved approximately **92% accuracy** on local validation sets and **84% on the Kaggle dataset** after tuning hyperparameters like resolution and the final dense layer size.


### Recurrent Neural Networks (RNN)

Several iterations of RNNs were tested to capture temporal dependencies:

* Prototype & V1: Basic **LSTM** models that initially suffered from either extreme overfitting (reaching 99.9% training accuracy but failing on new data) or a total lack of learning.
* V2 & V3: Increased complexity (3 LSTM layers) and regularization (L2, Dropout) were added. A non-scaled data approach in V2.2 showed the model finally beginning to learn.
* **V4 (Hybrid CNN-RNN)**: A new approach where the **CNN model's initial layers** acted as a feature extractor, feeding processed data into a **GRU** layer.


---

## Key Results & Findings

| Model | Local Accuracy | Kaggle Accuracy | Notes |
| --- | --- | --- | --- |
| **Standalone CNN** | ~92% | **84%** | Most reliable standalone performer. |
| **RNN V3** | ~75%  | **54%**  | Significant overfitting observed. |
| **Hybrid CNN-RNN** | ~93% | **77%** | Shows high potential for future tuning. |

* **Confusion Matrix Insight**: The models generally perform well but sometimes confuse **Rotateright** and **Scrolldown** gestures.
* **Overfitting**: This remained a persistent challenge throughout the project, particularly for pure RNN architectures.
