# Machine Learning — Concepts and Projects

This repository contains Python-based Machine Learning implementations covering core concepts and practical projects. All code is written using scikit-learn, pandas, and matplotlib.

---

## Repository Structure

```
MechineLearning/
|
|-- K-MeansClustering.py              # Unsupervised clustering with K-Means
|-- KFold.py                          # K-Fold cross-validation technique
|-- LogisticRegression.py             # Binary classification with Logistic Regression
|-- OneHotEncoding.py                 # Categorical encoding with One-Hot Encoder
|
`-- Projects/
    |-- HousePricePrediction/
    |   |-- HousePricePrediction.py   # Linear Regression on real estate data
    |   `-- data.csv                  # King County, WA housing dataset
    |
    `-- StudentMarksPrediction/
        `-- StudentMarksPrediction.py # Linear Regression on study hours vs marks
```

---

## Concepts Overview

### 1. K-Means Clustering

An unsupervised learning algorithm that groups unlabeled data into a specified number of clusters based on feature similarity. The model assigns each data point to the nearest cluster centroid and iteratively updates centroids until convergence.

**Refer to:** `K-MeansClustering.py`

---

### 2. K-Fold Cross-Validation

A model evaluation technique that splits the dataset into K equal folds. The model is trained K times, each time using a different fold as the test set and the remaining folds as the training set. This provides a more reliable estimate of model performance than a single train-test split.

**Refer to:** `KFold.py`

---

### 3. Logistic Regression

A supervised classification algorithm used to predict binary outcomes. Despite its name, it is a classification model that outputs probabilities using the sigmoid function and maps them to discrete class labels.

**Refer to:** `LogisticRegression.py`

---

### 4. One-Hot Encoding

A preprocessing technique for converting categorical text features into a numerical binary matrix. Each unique category becomes a separate column with a value of 0 or 1, making it compatible with ML algorithms that require numeric input.

**Refer to:** `OneHotEncoding.py`

---

## Projects Overview

### House Price Prediction

A regression project that predicts house prices using the King County, Washington real estate dataset. The pipeline includes data cleaning, feature engineering, categorical encoding, standard scaling, and Linear Regression modeling. Model performance is evaluated using R2 score and Mean Squared Error.

**Refer to:** `Projects/HousePricePrediction/README.md`

---

### Student Marks Prediction

A simple regression project that models the relationship between study hours and exam marks. It demonstrates the core workflow of building and visualizing a Linear Regression model on a small custom dataset.

**Refer to:** `Projects/StudentMarksPrediction/README.md`

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| Python 3.x | Core language |
| scikit-learn | ML algorithms and preprocessing |
| pandas | Data loading and manipulation |
| matplotlib | Visualization and plotting |

---

## Installation

```bash
git clone https://github.com/yourusername/MechineLearning.git
cd MechineLearning
pip install scikit-learn pandas matplotlib
```

Run any script directly:

```bash
python K-MeansClustering.py
python Projects/HousePricePrediction/HousePricePrediction.py
```

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Arpit Gangwar**
B.Tech Computer Science and Engineering
Invertis University, Bareilly
