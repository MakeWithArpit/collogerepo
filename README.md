# Machine Learning — Concepts and Projects

This repository contains Python-based Machine Learning implementations covering core preprocessing techniques, model evaluation methods, and practical end-to-end projects. All code is written using scikit-learn, pandas, and matplotlib.

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

## Files Overview

### 1. K-Means Clustering (`K-MeansClustering.py`)

Demonstrates the K-Means unsupervised clustering algorithm. A list of 8 numerical data points is grouped into 2 clusters using `KMeans(n_clusters=2)`. The model assigns a cluster label (0 or 1) to each data point based on proximity to the nearest centroid.

**Key concepts:** Unsupervised learning, centroids, cluster labels, inertia.

```python
model = KMeans(n_clusters=2)
model.fit(a)
print(model.labels_)
```

---

### 2. K-Fold Cross-Validation (`KFold.py`)

Demonstrates how K-Fold cross-validation splits a dataset into multiple training and testing subsets. Using 5 folds on a 5-element dataset, each data point takes a turn as the test set while the remaining 4 form the training set. This gives a more reliable performance estimate than a single train-test split.

**Key concepts:** Fold splitting, train-test indices, evaluation reliability.

```python
kf = KFold(n_splits=5)
for train, test in kf.split(data):
    print("Train:", [data[i] for i in train])
    print("Test:", [data[i] for i in test])
```

---

### 3. Logistic Regression (`LogisticRegression.py`)

Implements binary classification using Logistic Regression. A small dataset with 4 labeled samples is used to train a classifier that predicts the class (0 or 1) of a new input value. Despite the name, Logistic Regression is a classification model — not regression.

**Key concepts:** Binary classification, sigmoid function, decision boundary.

```python
model = LogisticRegression()
model.fit(X, Y)
print(model.predict([[40]]))
```

---

### 4. One-Hot Encoding (`OneHotEncoding.py`)

Demonstrates how to convert categorical text data (city names) into a numerical binary matrix using `OneHotEncoder`. Each unique category gets its own column with a value of 1 or 0. This avoids the false ordinal ranking that simple integer encoding would imply.

**Key concepts:** Categorical encoding, sparse matrix, dummy variables.

```python
encoder = OneHotEncoder()
result = encoder.fit_transform(city)
print(result.toarray())
```

---

### 5. House Price Prediction (`Projects/HousePricePrediction/`)

An end-to-end regression project using the King County, WA housing dataset. The pipeline covers data cleaning, date feature extraction, outlier removal, one-hot encoding of categorical columns (`city`, `statezip`), standard scaling, and Linear Regression modeling. Performance is evaluated using R2 score and Mean Squared Error. A scatter plot of actual vs predicted prices is generated at the end.

**Key concepts:** Full ML pipeline, feature engineering, StandardScaler, data leakage prevention, R2 score, MSE.

**Dataset columns:** `price`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `city`, `statezip`.

```python
data.drop(columns=["street", "country"], inplace=True)
data = pd.get_dummies(data, columns=["city", "statezip"], drop_first=True)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
model = LinearRegression().fit(X_train, y_train)
```

---

### 6. Student Marks Prediction (`Projects/StudentMarksPrediction/`)

A beginner-level regression project that models the relationship between study hours and exam marks on a small custom dataset. A Linear Regression model is trained, a prediction is made for 8 hours of study, and the data along with the regression line is plotted using matplotlib.

**Key concepts:** Linear Regression, prediction on unseen input, scatter plot with regression line.

```python
model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[8]])
print("Marks of 8 hours study:", prediction[0])
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| Python 3.x | Core language |
| scikit-learn | ML algorithms, preprocessing, evaluation |
| pandas | Data loading and manipulation |
| matplotlib | Visualization and plotting |

---

## Installation and Setup

```bash
git clone https://github.com/yourusername/MechineLearning.git
cd MechineLearning
pip install scikit-learn pandas matplotlib
```

Run any concept script:

```bash
python K-MeansClustering.py
python KFold.py
python LogisticRegression.py
python OneHotEncoding.py
```

Run a project:

```bash
cd Projects/HousePricePrediction
python HousePricePrediction.py

cd Projects/StudentMarksPrediction
python StudentMarksPrediction.py
```

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Arpit Gangwar**
B.Tech Computer Science and Engineering
Invertis University, Bareilly
