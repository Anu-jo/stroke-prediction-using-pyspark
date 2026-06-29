# 🧠 Stroke Prediction Using PySpark Machine Learning

## 📌 Project Overview

This project develops a scalable machine learning pipeline using **PySpark** to predict the likelihood of stroke occurrence based on patient demographic and clinical information.

The project demonstrates the complete machine learning workflow, including data preprocessing, feature engineering, dimensionality reduction, and model development using **Logistic Regression**. PySpark was used to efficiently process healthcare data and prepare it for predictive modeling.

---

## 🎯 Objectives

- Analyze healthcare data using PySpark.
- Clean and preprocess the dataset.
- Convert data into appropriate formats for machine learning.
- Perform feature engineering and dimensionality reduction.
- Build a Logistic Regression model for stroke prediction.
- Evaluate the effectiveness of the prediction model.

---

## 📊 Dataset

This project uses the **Stroke Prediction Dataset** from Kaggle.

### Dataset Information

- **Total Records:** 5,110
- **Features:** 12
- **Target Variable:** Stroke

### Features

| Feature | Description |
|----------|-------------|
| gender | Patient gender |
| age | Age |
| hypertension | Hypertension status |
| heart_disease | Heart disease status |
| ever_married | Marital status |
| work_type | Employment type |
| Residence_type | Urban/Rural |
| avg_glucose_level | Average glucose level |
| bmi | Body Mass Index |
| smoking_status | Smoking history |
| stroke | Target variable (0 = No Stroke, 1 = Stroke) |

**Dataset Source**

https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

---

## 🛠 Technologies Used

- Python
- PySpark
- Apache Spark
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📂 Repository Structure

```text
```text
stroke-prediction-using-pyspark/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── healthcare-dataset-stroke-data.csv
│   └── stroke_dataset.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── datatype_changing.py
│   ├── dimensionality_reduction.py
│   └── logistic_regression.py
