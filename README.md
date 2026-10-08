# First Machine Learning Classifier

A machine learning project using Scikit-Learn to train and evaluate a Decision Tree Classifier.

## Overview

This project demonstrates an end-to-end machine learning classification pipeline. It handles dataset loading, train-test splitting, model training, depth control, and model evaluation.

## Features

- **Data Pipeline:** Loads standard dataset and structures features using Pandas.
- **Train-Test Split:** Divides data into 80% training and 20% testing sets using stratified sampling.
- **Model Training:** Builds a Decision Tree Classifier with maximum depth constraints to prevent overfitting.
- **Model Evaluation:** Computes accuracy score, confusion matrix, precision, recall, and F1 score.
- **Interpretability:** Generates feature importance rankings and human-readable decision rules.

## Tech Stack

- Python
- Scikit-Learn
- Pandas
- NumPy

## Execution Instructions

Run the script directly in Google Colab or locally via command line:

```bash
python ml_classifier.py
