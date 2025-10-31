# ♻️ Garbage Classifier: Waste Type Recognition

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/code/saqibshoaibdz/trash-and-waste-type-recognition)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-green?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

This project implements a deep learning model, specifically a **Convolutional Neural Network (CNN)**, to classify images of garbage into 13 distinct categories. The goal is to provide an automated solution for waste sorting, which is critical for efficient recycling and environmental management.

This repository contains the code and methodology, while the trained model and analysis are detailed in the accompanying **[Kaggle Notebook](https://www.kaggle.com/code/saqibshoaibdz/trash-and-waste-type-recognition)**.

## 📖 Table of Contents

* [Project Overview](#-project-overview)
* [Problem Statement](#-problem-statement)
* [Dataset](#-dataset)
* [Model Architecture](#-model-architecture)
* [Methodology & Training](#-methodology--training)
* [Performance & Evaluation](#-performance--evaluation)
* [Technology Stack](#-technology-stack)
* [How to Use](#-how-to-use)
* [File Structure](#-file-structure)
* [License](#-license)
* [Acknowledgments](#-acknowledgments)

---

## 🔬 Project Overview

This project leverages computer vision to tackle the real-world challenge of waste management. By building a multi-class image classifier, we can automatically identify the type of trash from an image. The model is built from scratch using the TensorFlow and Keras libraries and trained on a public dataset containing over 17,000 images.

## 🎯 Problem Statement

Improper waste segregation is a significant global issue, leading to polluted landfills, contaminated recycling streams, and inefficient resource recovery. Manual sorting is slow, expensive, and hazardous. An automated system can:
* Improve the speed and accuracy of waste sorting.
* Increase recycling rates and purity.
* Reduce manual labor costs and health risks.
* Power smart-bins and automated sorting facilities.

---

## 📊 Dataset

This model is trained on the **[Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)** dataset from Kaggle.

* **Source:** Kaggle
* **Size:** 17,750 images
* **Classes:** 13 distinct categories
* **Image Properties:** Varying sizes and resolutions.

The 13 classes included in the dataset are:
1.  Battery
2.  Biological
3.  Brown-Glass
4.  Cardboard
5.  Clothes
6.  E-Waste
7.  Green-Glass
8.  Metal
9.  Paper
10. Plastic
11. Shoes
12. Trash (General)
13. White-Glass

---

## 🏗️ Model Architecture

The classifier is a **Sequential Convolutional Neural Network (CNN)** built with the Keras API. A CNN is ideal for image classification tasks as it can automatically learn hierarchical features from the images.



[Image of a convolutional neural network architecture]


The architecture consists of:
1.  **Input Layer:** `(224, 224, 3)` - Accepts 224x224 pixel RGB images.
2.  **Convolutional Block 1:**
    * `Conv2D` (32 filters, 3x3 kernel, 'relu' activation)
    * `MaxPooling2D` (2x2 pool size)
3.  **Convolutional Block 2:**
    * `Conv2D` (64 filters, 3x3 kernel, 'relu' activation)
    * `MaxPooling2D` (2x2 pool size)
4.  **Convolutional Block 3:**
    * `Conv2D` (128 filters, 3x3 kernel, 'relu' activation)
    * `MaxPooling2D` (2x2 pool size)
5.  **Convolutional Block 4:**
    * `Conv2D` (256 filters, 3x3 kernel, 'relu' activation)
    * `MaxPooling2D` (2x2 pool size)
6.  **Flatten Layer:** Flattens the 4D tensor output from the conv blocks into a 1D vector.
7.  **Dense Block:**
    * `Dense` (256 units, 'relu' activation) - A fully connected hidden layer.
    * `Dropout` (0.5) - A regularization technique to prevent overfitting.
8.  **Output Layer:**
    * `Dense` (13 units, 'softmax' activation) - Outputs a probability distribution across the 13 classes.

---

## ⚙️ Methodology & Training

### 1. Data Preprocessing & Augmentation
* **`ImageDataGenerator`:** The Keras `ImageDataGenerator` is used for preprocessing and real-time data augmentation.
* **Splitting:** The dataset is split into 80% for training and 20% for validation.
* **Rescaling:** All pixel values are normalized from `[0, 255]` to `[0, 1]` by dividing by 255.
* **Augmentation:** To improve model generalization and prevent overfitting, the following augmentations are applied to the training set:
    * `shear_range=0.2`
    * `zoom_range=0.2`
    * `horizontal_flip=True`
    * `vertical_flip=True`

### 2. Model Compilation
* **Optimizer:** **Adam** - An adaptive learning rate optimizer that is well-suited for a wide range of problems.
* **Loss Function:** **Categorical Crossentropy** - Standard loss function for multi-class classification problems.
* **Metrics:** **Accuracy** - The primary metric for evaluating model performance.

### 3. Training
* **Epochs:** The model is trained for **50 epochs**.
* **Batch Size:** 32
* **Callbacks:**
    * **`EarlyStopping`:** Monitors the `val_loss` and stops training after 5 epochs of no improvement, preventing overfitting and saving compute time.
    * **`ReduceLROnPlateau`:** Monitors the `val_loss` and reduces the learning rate by a factor of 0.1 if no improvement is seen for 3 epochs.

---

## 📈 Performance & Evaluation

The model's performance is evaluated using the 20% validation set.

* **Training & Validation Plots:** The training history is visualized to check for overfitting by plotting **Accuracy vs. Epochs** and **Loss vs. Epochs** for both training and validation sets.
* **Classification Report:** A detailed report from `sklearn.metrics` is generated, showing the **precision, recall, and f1-score** for each of the 13 classes.
* **Confusion Matrix:** A heatmap is generated using `seaborn` to visualize the model's performance on a per-class basis. This helps identify which classes are commonly confused (e.g., 'brown-glass' vs. 'green-glass').

*(For detailed plots and final accuracy/loss metrics, please see the [Kaggle Notebook](https://www.kaggle.com/code/saqibshoaibdz/trash-and-waste-type-recognition).)*

---

## 💻 Technology Stack

* **Core:** Python 3
* **Deep Learning:** TensorFlow, Keras
* **Data Manipulation:** NumPy, Pandas
* **Data Visualization:** Matplotlib, Seaborn
* **Metrics:** Scikit-learn (sklearn)
* **Environment:** Kaggle Notebooks

---

## 🚀 How to Use

To run this project on your local machine, follow these steps.

### 1. Prerequisites
* Python 3.8+
* `pip` and `venv` (or `conda`)

### 2. Clone the Repository
```bash
git clone [https://github.com/dot-css/Garbage-Classifier.git](https://github.com/dot-css/Garbage-Classifier.git)
cd Garbage-Classifier
````

### 3\. Set Up a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n garbage-classifier python=3.9
conda activate garbage-classifier
```

### 4\. Install Dependencies

Create a `requirements.txt` file with the following content, then run `pip install`:

```
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
```

```bash
pip install -r requirements.txt
```

### 5\. Download the Dataset

1.  Go to the [Kaggle Dataset page](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2).
2.  Download the dataset (e.g., `archive.zip`).
3.  Unzip the contents into a `data/` directory within the project folder. Your structure should look like:
    ```
    Garbage-Classifier/
    ├── data/
    │   ├── battery/
    │   ├── biological/
    │   └── ... (all 13 class folders)
    ├── trash-and-waste-type-recognition.ipynb
    └── README.md
    ```

### 6\. Run the Notebook

Launch Jupyter Notebook or Jupyter Lab and open `trash-and-waste-type-recognition.ipynb`.

```bash
jupyter notebook
```

You can now run the cells in the notebook to load the data, train the model, and evaluate its performance.

-----

## 🌳 File Structure

```
.
├── data/                       # Contains the dataset (must be downloaded)
│   ├── battery/
│   ├── biological/
│   ├── ... (and so on for 13 classes)
├── trained_model/              # (Optional) Place to save the final .h5 model
│   └── garbage_classifier.h5
├── trash-and-waste-type-recognition.ipynb  # The main Jupyter Notebook
├── requirements.txt            # List of Python dependencies
└── README.md                   # This file
```

-----

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/licenses/MIT) file for details.

-----

## 🙏 Acknowledgments

  * **Dataset:** Credit to the creator of the [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) dataset on Kaggle.
  * **Inspiration:** This project is inspired by the need for better, automated solutions in waste management and recycling.

<!-- end list -->

```
```
