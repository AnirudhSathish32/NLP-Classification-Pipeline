# Production-Ready NLP Classification Pipeline

## Overview
A modular NLP system that classifies text using classical ML and a lightweight neural network.

## Why NLP Classification Matters
Text classification powers sentiment analysis, moderation systems, support routing, and recommendation filtering.

## Architecture

Raw Data → Cleaning → TF-IDF → Models → CV → Evaluation → Model Selection → Saved Pipeline

## Models Compared
- Logistic Regression
- Linear SVM
- Random Forest
- MLPClassifier (Neural Network)

## Metrics
- Precision
- Recall
- F1
- ROC-AUC
- Confusion Matrix

ROC-AUC is emphasized because it measures ranking quality under class imbalance.

## How to Run

```bash
docker compose up --build

## Dataset

https://ai.stanford.edu/~amaas/data/sentiment/
