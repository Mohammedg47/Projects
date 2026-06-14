# Machine Learning Projects

A collection of Python-based machine learning projects built while learning core ML concepts, including regression, gradient descent, and data preprocessing. Each project uses real datasets and is implemented with NumPy, pandas, and scikit-learn.

---

## Projects

### 1. Housing Price Prediction
**File:** `housing_prices.py` | **Data:** `houses.txt`

Predicts housing prices from four features — square footage, number of bedrooms, number of floors, and age of the home. Uses scikit-learn's `SGDRegressor` with z-score normalization via `StandardScaler` to handle the wide range of feature scales. Includes scatter plots comparing predicted vs. actual prices across each feature.

**Key concepts:** multi-variable linear regression, feature normalization, stochastic gradient descent

---

### 2. Salary Progression Predictor
**File:** `salary_progression.py` | **Data:** `Salary_dataset.csv`

Predicts salary from years of experience using a single-variable linear regression model built from scratch. Implements the cost function and gradient descent manually without any ML libraries, demonstrating the underlying math behind model training.

**Key concepts:** univariate linear regression, cost function, gradient descent from scratch

---

### 3. Student Performance Predictor
**File:** `student_performance.py` | **Data:** `Student_Performance.csv`

Predicts a student's academic performance index from study habits and lifestyle factors (hours studied, previous scores, sleep hours, and practice papers completed). Implements multi-variable linear regression and batch gradient descent from scratch with a 75/25 train/test split. Includes visualizations of the normalization process and predicted vs. actual results.

**Key concepts:** multi-variable linear regression, train/test split, z-score normalization, batch gradient descent

---

### 4. Counterfeit Banknote Detector
**File:** `fake.py` | **Data:** `fake_bills.csv`

Analyzes physical measurements of banknotes (diagonal, margins, height, length) to distinguish genuine bills from counterfeits. Adds engineered features (left and right surface area) and visualizes how normalization affects feature distributions for both classes.

**Key concepts:** feature engineering, data visualization, StandardScaler normalization, exploratory data analysis

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** NumPy, pandas, scikit-learn, Matplotlib

## Author

Mohammed — Electrical Engineering student at NC State University (Class of 2027)
