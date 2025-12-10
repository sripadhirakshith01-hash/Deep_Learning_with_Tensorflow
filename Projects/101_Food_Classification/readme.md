# 🍽️ **Food-101 Classification using EfficientNetV2B0**

*A Deep Learning approach for large-scale food image recognition*


---

## 📌 **Overview**

This project implements a **Food-101 image classification model** using **EfficientNetV2B0**, leveraging both **feature extraction** and **fine-tuning** to achieve strong generalization on diverse food categories.
The model is built using **TensorFlow/Keras** and trained under computational constraints while still achieving **high validation performance**.

---

## 📝 **Abstract**

Food recognition plays a crucial role in dietary analysis, nutrition tracking, and smart restaurant systems.
In this study, a **deep learning computer vision model** was trained on the **Food-101 dataset** using **EfficientNetV2B0**. Techniques such as **data augmentation, early stopping, batch normalization, dropout, LR scheduling, and AdamW optimizer** were incorporated to prevent overfitting.

📈 **Final Performance**

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 64%    |
| Validation Accuracy | 76–77% |

The results demonstrate the effectiveness of modern CNN architectures for large-scale food image classification.


---

# 📚 **Table of Contents**

* [Introduction](#-introduction)
* [Related Work](#-related-work)
* [Dataset: Food-101](#-dataset-food101)
* [Model Architecture](#-model-architecture)
* [Training Techniques](#-training-techniques)
* [Training Configuration](#-training-configuration)
* [Experiments & Results](#-experiments--results)
* [Discussion](#-discussion)
* [Conclusion & Future Work](#-conclusion--future-work)
* [References](#-references)

---

# 🔍 **Introduction**

Food image classification is widely used in:

* 🍱 dietary assessment
* 🏥 healthcare analytics
* 🍽️ smart restaurants
* 📱 automated nutrition apps

With advancements in **CNNs and transfer learning**, pretrained models such as **EfficientNetV2** significantly improve performance while reducing computational cost.


---

# 📖 **Related Work**

| Study                           | Approach                        | Accuracy                   |
| ------------------------------- | ------------------------------- | -------------------------- |
| **Bossard et al., 2014**        | Food-101 with Random Forests    | ~50.76%                    |
| **Liu et al., DeepFood (2016)** | Deep CNN for dietary assessment | Top-1: 77.4%, Top-5: 93.7% |

Modern architectures continue improving efficiency and accuracy, motivating the use of EfficientNetV2 in this project.


---

# 🍔 **Dataset: Food-101**

The **Food-101** dataset contains:

| Property                  | Value                 |
| ------------------------- | --------------------- |
| Total Images              | 101,000               |
| Total Classes             | 101                   |
| Training Images per Class | 750                   |
| Test Images per Class     | 250                   |
| Preprocessing             | Max side length 512px |

⚠️ Training images intentionally include noise (bad lighting, incorrect labels), making the dataset challenging.


---

# 🧠 **Model Architecture — EfficientNetV2B0**

### ⭐ Why EfficientNetV2B0?

* Excellent **accuracy vs. efficiency** balance
* Lightweight → works on limited hardware
* Compound scaling (depth/width/resolution)
* Strong performance with **transfer learning**

### 🏗️ Architecture Flow (Simplified)

```
Input (224x224)
        │
EfficientNetV2B0 Backbone (Frozen initially)
        │
Global Average Pooling
        │
Dropout / BatchNorm
        │
Dense Layer (Classification Head)
        │
Softmax (101 classes)
```

### 🔄 Two-Phase Training

| Phase                     | Description                                 |
| ------------------------- | ------------------------------------------- |
| **1. Feature Extraction** | Freeze base model → train only dense layers |
| **2. Fine-Tuning**        | Unfreeze ~120 layers → train with lower LR  |

This two-stage strategy allows general ImageNet features to adapt to Food-101’s fine-grained classes.


---

# ⚙️ **Training Techniques**

To stabilize training and avoid overfitting, the following methods were used:

| Technique                      | Purpose                                               |
| ------------------------------ | ----------------------------------------------------- |
| 🎨 **Data Augmentation**       | Improve generalization via rotation, zoom, flip, etc. |
| ⏹️ **Early Stopping**          | Prevent overfitting when validation loss stagnates    |
| 📊 **Batch Normalization**     | Faster convergence + stable gradients                 |
| 📉 **Learning Rate Scheduler** | Reduce LR as training progresses                      |
| ⚙️ **AdamW Optimizer**         | Better regularization via weight decay                |



---

# 🧪 **Training Configuration**

| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Image Size        | 224 × 224                       |
| Batch Size        | 32                              |
| Epochs            | ~30 (Early Stopping enabled)    |
| Loss              | Categorical Crossentropy        |
| Training Strategy | Transfer Learning + Fine-Tuning |

### 💡 Two-Phase Strategy

| Phase                                                | Purpose                        |
| ---------------------------------------------------- | ------------------------------ |
| **Phase 1:** Freeze backbone → train classifier head | Fast, stable training          |
| **Phase 2:** Unfreeze layers → fine-tune             | Learn domain-specific features |



---

# 📊 **Experiments & Results**

### 🧪 Experiment Strategy

Because of GPU limits:

* Start with **10% of dataset**
* Scale up gradually after verifying model stability
* Compare multiple architectures
* Final choice: ✔️ **EfficientNetV2B0**

### 📈 Final Performance

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 64%    |
| Validation Accuracy | 76–77% |

### 🔍 Observations (Before Fine-Tuning)

* Base model frozen → only classifier learned
* Training accuracy plateau ~50%
* Validation accuracy > training accuracy (common in transfer learning)
* Underfitting due to frozen convolutional layers

### 🔍 Observations (After Fine-Tuning)

* Unfreezing top ~120 layers → large accuracy boost
* Training accuracy rose significantly
* Validation accuracy reached **76–77%**
* Training slower but more effective
* Regularization prevented overfitting



---

# 🗣️ **Discussion**

EfficientNetV2B0 provided an excellent trade-off between **efficiency and accuracy**, making it ideal for resource-limited training environments.
Misclassifications were mainly between visually similar food categories (e.g., pasta dishes, grilled foods). Noise and cluttered backgrounds in Food-101 remain challenging even for modern architectures.

The study highlights how:

* model depth
* optimization techniques
* computational constraints

all influence real-world deep learning performance.


---

# 🚀 **Conclusion & Future Work**

### ✅ Current Achievements

* Built a strong baseline with **EfficientNetV2B0**
* Achieved **76–77% validation accuracy**
* Demonstrated effective use of **transfer learning + fine-tuning**
* Showed benefits of augmentation & regularization

### 🔮 Future Work

* Test deeper EfficientNetV2 variants (B1–B3)
* Build a **Food Recommendation System**
* Deploy web app using **Streamlit or Gradio**
* Further **hyperparameter tuning**



---

# 📚 **References**

(All references sourced from your research paper.)


1. Bossard et al. (2014) — *Food-101 Dataset & Random Forests*
2. Liu et al. (2016) — *DeepFood*
3. Sahoo et al. (2019) — *FoodAI*
4. Kaggle Food-101 Dataset (DanB, 2017)
5. mrdbourke — *Food Vision Project*

