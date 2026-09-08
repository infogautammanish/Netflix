# Netflix
Machine learning project that classifies Netflix titles as Movies or TV Shows using Python, Pandas, Scikit-learn, feature engineering, preprocessing pipelines, model comparison, cross-validation, and hyperparameter tuning.


Netflix Content Classification using Machine Learning

A supervised machine learning project that classifies Netflix titles as Movie or TV Show using metadata from the Netflix Titles dataset.

📌 Project Overview

The goal of this project is to build and evaluate binary classification models that predict the content type of a Netflix title.

Machine Learning Type: Supervised Learning
Problem Type: Binary Classification
Dataset: Netflix Titles Dataset
Dataset Size: 8,807 rows × 12 columns

🎯 Objective

Build a machine learning classification pipeline that can:

Understand and explore Netflix title metadata

Clean missing and duplicate data

Engineer useful features from the duration column

Preprocess numerical and categorical features

Train multiple classification algorithms

Compare model performance

Analyze feature importance

Make predictions for new Netflix titles

Save and reload the trained model

Validate performance using cross-validation

Tune Random Forest hyperparameters using GridSearchCV

🗂️ Project Workflow

Import Libraries
      ↓
Load Dataset
      ↓
Data Understanding
      ↓
Exploratory Data Analysis
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Train-Test Split
      ↓
Preprocessing Pipeline
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Comparison
      ↓
Feature Importance
      ↓
New Predictions
      ↓
Model Saving & Loading
      ↓
Cross-Validation
      ↓
Hyperparameter Tuning

📊 Dataset

The project uses the Netflix Titles Dataset containing information about Netflix movies and TV shows.

The original dataset contains these 12 columns:

show_id

type

title

director

cast

country

date_added

release_year

rating

duration

listed_in

description

Target Variable

The target variable is:

type

Possible classes:

Movie

TV Show

The dataset distribution in the notebook is:

Content Type

Percentage

Movie

69.62%

TV Show

30.38%

🔎 Exploratory Data Analysis

The notebook explores:

Movie vs TV Show distribution

Content type percentages

Release year distribution

Netflix ratings distribution

Missing-value patterns

Missing Values

The largest missing-value counts in the original dataset include:

Column

Missing Values

Missing %

director

2,634

29.91%

country

831

9.44%

cast

825

9.37%

date_added

10

0.11%

rating

4

0.05%

duration

3

0.03%

🧹 Data Cleaning

For the basic classification workflow, the following columns are removed:

show_id
title
director
cast
description
date_added

The project also checks for duplicate records.

🛠️ Feature Engineering

The duration column contains values such as:

90 min
2 Seasons
1 Season

Two numerical/categorical features are extracted:

duration_value

The numerical value is extracted from the original duration.

Examples:

90 min   → 90
2 Seasons → 2

duration_type

The duration is categorized as:

Minutes
Seasons

The original duration column is then removed.

🤖 Machine Learning Models

Three classification algorithms are trained and evaluated:

Logistic Regression

Decision Tree

Random Forest

The project uses a preprocessing pipeline with ColumnTransformer, including:

Missing-value imputation

One-hot encoding for categorical variables

Standard scaling for numerical variables

This keeps preprocessing consistent between training and prediction.

📈 Model Evaluation

The notebook evaluates models using:

Accuracy

Precision

Recall

F1 Score

Classification Report

Confusion Matrix

Results

The notebook reports the following test-set results:

Model

Accuracy

Precision

Recall

F1 Score

Logistic Regression

1.0000

1.0000

1.0000

1.0000

Decision Tree

1.0000

1.0000

1.0000

1.0000

Random Forest

1.0000

1.0000

1.0000

1.0000

The notebook selects Logistic Regression as the best model based on the model-comparison table, where the first model with the highest F1 score is selected.

🌲 Random Forest Analysis

Random Forest is additionally analyzed for feature importance.

The notebook uses the trained Random Forest model to investigate which processed features contribute to classification.

🔮 Example Prediction

The notebook demonstrates prediction on a new title with metadata such as:

Country: India
Release Year: 2024
Rating: TV-MA
Listed In: International TV Shows, TV Dramas
Duration Value: 2
Duration Type: Seasons

The trained Random Forest classifier is used to predict whether the title is a:

Movie

or

TV Show

💾 Model Saving

The trained model is saved using joblib:

netflix_content_classifier.pkl

It can later be loaded and reused without retraining:

loaded_model = joblib.load("netflix_content_classifier.pkl")

🔁 Cross-Validation

The project uses 5-fold Stratified Cross-Validation with weighted F1 scoring.

Notebook result:

Cross-Validation F1 Scores:
[1. 1. 1. 1. 1.]

Mean F1 Score: 1.0000
Std Deviation: 0.0000

This indicates perfectly consistent performance across the five reported folds in the notebook.

⚙️ Hyperparameter Tuning

Random Forest is further optimized using GridSearchCV.

The reported best parameters are:

{
    "classifier__max_depth": 10,
    "classifier__min_samples_leaf": 1,
    "classifier__min_samples_split": 2,
    "classifier__n_estimators": 100
}

Reported best cross-validation F1 score:

1.0000

🧰 Technologies Used

Programming Language

Python

Data Analysis

Pandas

NumPy

Data Visualization

Matplotlib

Seaborn

Machine Learning

Scikit-learn

Model Persistence

Joblib

Models

Logistic Regression

Decision Tree

Random Forest

Notebook Environment

Jupyter Notebook

📁 Project Structure

A recommended repository structure is:

netflix-content-classification/
│
├── Netflix.ipynb
├── netflix_titles.csv
├── netflix_content_classifier.pkl
├── README.md
└── requirements.txt

🚀 How to Run

1. Clone the repository

git clone <your-repository-url>
cd netflix-content-classification

2. Install dependencies

pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter

Or, if a requirements.txt file is included:

pip install -r requirements.txt

3. Start Jupyter Notebook

jupyter notebook

Open:

Netflix.ipynb

4. Add the Dataset

Make sure:

netflix_titles.csv

is available in the same working directory as the notebook.

5. Run the Notebook

Run the cells from top to bottom to reproduce the analysis, training, evaluation, cross-validation, and hyperparameter tuning.

⚠️ Important Modeling Note

The notebook reports 1.0000 performance across the evaluated models and cross-validation. However, this result should be interpreted carefully.

The target is type (Movie vs TV Show), while the engineered duration_type is derived directly from the original duration field. The duration format itself contains information such as min versus Season, which is closely tied to the target class.

Therefore, the perfect scores may indicate target leakage or a feature that effectively reveals the target, rather than demonstrating that the model can generalize perfectly to unseen Netflix content.

For a stronger real-world version of this project, remove target-revealing features and re-evaluate the models.

📌 Key Learnings

This project demonstrates an end-to-end classification workflow:

Loading and inspecting real-world data

Handling missing values

Exploratory data analysis

Feature engineering

Categorical and numerical preprocessing

Building reusable ML pipelines

Training multiple classification models

Comparing model performance

Confusion matrix analysis

Feature importance analysis

Making predictions on new data

Saving and loading trained models

Cross-validation

Hyperparameter tuning

🔮 Future Improvements

Possible extensions include:

Remove potentially target-leaking features and rebuild the models

Add stronger feature engineering from Netflix metadata

Compare additional algorithms such as XGBoost and SVM

Build a Streamlit prediction interface

Add automated model evaluation

Track experiments and metrics

Deploy the trained model as a web application

Add unit tests and a production-ready inference pipeline

👨‍💻 Project

Netflix Content Classification using Machine Learning

Built as an end-to-end machine learning classification project using Python and Scikit-learn.
