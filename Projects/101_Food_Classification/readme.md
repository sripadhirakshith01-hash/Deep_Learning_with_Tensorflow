# 🍽️ Food-101 Large-Scale Classification using EfficientNetV2B0

## 🚀 Project Overview

This repository presents a robust deep learning model for classifying 101 distinct food categories from the challenging **Food-101 dataset**. By employing **EfficientNetV2B0** and a meticulous two-phase transfer learning strategy, we achieved high predictive performance while maintaining computational efficiency—a critical balance for real-world deployment.

### 🎯 Key Performance Metric

| Metric | Value |
| :--- | :--- |
| **Final Validation Accuracy** | $\mathbf{77.0\%}$ |
| **Model** | EfficientNetV2B0 (Pretrained) |
| **Strategy** | Transfer Learning + Fine-Tuning |


## 1\. Motivation

Food image classification is a cornerstone of digital health and smart systems, facilitating automated dietary assessment, nutrition tracking, and smart restaurant inventory. The goal was to develop an accurate model that minimizes computational resource demand, making **EfficientNetV2V0**—known for its superior accuracy-to-latency trade-off—the ideal choice.

-----

## 2\. The Food-101 Dataset

The **Food-101** dataset is a benchmark for fine-grained classification, comprising 101,000 images across 101 categories. Its complexity stems from **high visual similarity** between certain classes (e.g., various pasta types) and the presence of **noisy labels and backgrounds**.

| Property | Value |
| :--- | :--- |
| **Total Images** | $\mathbf{101,000}$ |
| **Total Classes** | $\mathbf{101}$ |
| **Data Split** | 750 Train Images / 250 Test Images per class |

-----

## 3\. Model Architecture: EfficientNetV2B0

### **Why EfficientNetV2B0?**

The EfficientNetV2 family represents a breakthrough in neural network scaling. They utilize an optimized combination of scaling dimensions (depth, width, and resolution) and integrate the more efficient **Fused-MBConv** structure in the initial layers, leading to faster training and better parameter utilization than its predecessors.

### **Architecture Flow**

The base model, pretrained on ImageNet, is used as a powerful feature extractor. A custom classification head is appended for the 101-class problem:

$$\text{Model} = \text{EfficientNetV2B0}_{\text{(Base)}} \to \text{GlobalAveragePooling} \to \text{Dropout} \to \text{Dense}(\text{101 classes}) \to \text{Softmax}$$

-----

## 4\. Methodology: Two-Phase Training

To ensure stability and optimal adaptation of the pretrained weights, a structured two-phase strategy was employed:

| Phase | Description | Goal | Learning Rate (LR) |
| :--- | :--- | :--- | :--- |
| **1. Feature Extraction (Warmup)** | Base model frozen; only classification head trained. | Rapidly converge the head layer to a feature-rich representation. | High ($\approx 1\times 10^{-3}$) |
| **2. Fine-Tuning (Adaptation)** | Top $\approx 120$ base layers unfrozen and trained with the head. | Adapt ImageNet-learned features to the subtle food specifics. | Low ($\approx 1\times 10^{-5}$) |

-----

## 5\. Training Configuration & Regularization

Training utilized TensorFlow/Keras on a resource-constrained GPU environment. Robust regularization was key to achieving high generalization on the validation set.

### 🛠️ Configuration

| Parameter | Value |
| :--- | :--- |
| **Input Resolution** | $224 \times 224$ |
| **Batch Size** | 32 |
| **Optimizer** | **AdamW** (with Weight Decay) |
| **Loss** | Categorical Crossentropy |

### 🛡️ Key Regularization Techniques

  * **Data Augmentation:** On-the-fly transformations (rotation, zoom, flip) to prevent overfitting to specific image layouts.
  * **Early Stopping:** Monitored validation loss to halt training when improvement plateaued.
  * **Learning Rate Scheduler:** Applied a decay schedule to ensure smooth convergence during the sensitive fine-tuning phase.

-----

## 6\. Results and Analysis

### 📈 Performance Metrics

| Metric | Value | Comment |
| :--- | :--- | :--- |
| **Training Accuracy** | $64\%$ | Lower than Validation, common post-augmentation, indicating good generalization. |
| **Validation Accuracy** | $\mathbf{76.0-77.0\%}$ | Competitive performance on a highly complex dataset. |

### **Observation**

The most significant performance boost ($\approx 12\%$) was observed during **Phase 2 (Fine-Tuning)**, confirming that adapting the top convolutional layers of EfficientNetV2B0 to the Food-101 domain was crucial for distinguishing fine-grained food classes.

-----

## 7\. Getting Started (Setup)

### **Prerequisites**

  * Python 3.8+
  * GPU access recommended for faster training.

### **Installation**

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YourUsername/Food-101-EfficientNetV2.git
    cd Food-101-EfficientNetV2
    ```

2.  **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    .\venv\Scripts\activate   # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### **Running the Model**

1.  **Download the Food-101 Dataset:** The dataset can be automatically downloaded and prepared using the `tensorflow_datasets` library as used in the training scripts.
2.  **Execute the training script:**
    ```bash
    python train_efficientnetv2.py --epochs 30 --batch_size 32
    ```
    *(Note: Refer to the specific training script for exact command-line arguments.)*

-----

## 8\. Future Work

The following enhancements are planned to build upon this robust baseline:

1.  **Model Exploration:** Experiment with deeper EfficientNetV2 variants (B1, B2) to assess the diminishing returns of increased model complexity.
2.  **Hyperparameter Optimization:** Conduct a comprehensive search for optimal learning rate and weight decay schedules using tools like Weights & Biases or TensorBoard.
3.  **Deployment & Inference:** Develop a lightweight web application (e.g., using Streamlit) for real-time inference demonstrations.
4.  **Ensembling:** Investigate model ensembling with other high-performing architectures (e.g., ConvNeXt) to potentially achieve state-of-the-art results.

-----

**License:** This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
