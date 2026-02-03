📰 Fake News Detection using Machine Learning

📌 Project Overview

This project focuses on detecting fake news articles using Natural Language Processing (NLP) and Machine Learning techniques.
The model classifies a given news article as Fake News or Real News based on textual patterns learned from a labeled dataset.

The goal of this project is classification based on language patterns, not fact verification.

🎯 Problem Statement

With the rapid spread of misinformation online, it is important to automatically identify fake news articles.
This project uses TF-IDF vectorization and Logistic Regression to classify news content efficiently.

📂 Dataset Description

Dataset name: news_dataset.csv

Source: Public fake news dataset (used for academic purposes)

Format: CSV

📊 Dataset Columns
Column Name	Description
text	The main content of the news article
label	News category (0 = Fake, 1 = Real)
✅ Dataset Characteristics

No missing values

Balanced or near-balanced class distribution

Contains strong linguistic patterns distinguishing fake and real news

⚠️ Note: The dataset contains stylistic and linguistic biases, which can lead to high accuracy scores.

🛠️ Technologies Used

Python

Pandas & NumPy – data handling

Scikit-learn – ML models and evaluation

TF-IDF Vectorizer – text feature extraction

Matplotlib & Seaborn – visualization

🔄 Project Workflow

Import required libraries

Load and inspect the dataset

Clean and preprocess text data

Split data into training and testing sets

Convert text into numerical features using TF-IDF

Train a Logistic Regression classifier

Evaluate model performance

Predict custom news input

🧹 Text Preprocessing

The following preprocessing steps are applied:

Convert text to lowercase

Remove URLs, HTML tags, punctuation, digits, and extra whitespace

Remove stopwords during TF-IDF vectorization

This helps reduce noise and improve model performance.

🔢 Feature Extraction (TF-IDF)

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical features based on word importance.

Key parameters:

stop_words='english'

max_df=0.7

min_df=2

🤖 Model Used
Logistic Regression

Well-suited for text classification

Fast and interpretable

Performs well with TF-IDF features

LogisticRegression(max_iter=1000)

📈 Model Evaluation

The model is evaluated using:

Accuracy Score

Confusion Matrix

Precision, Recall, and F1-score

⚠️ About High Accuracy

The model achieves ~99% accuracy, which is primarily due to:

Strong textual patterns in the dataset

Clean separation between fake and real news language

High accuracy does not imply factual correctness.
The model classifies news based on linguistic similarity, not truth verification.

📊 Visualization

A confusion matrix heatmap is used to visualize model performance and class-wise predictions.

🚀 How to Run the Project

Clone the repository

git clone https://github.com/your-username/fake-news-detection.git


Install dependencies

pip install -r requirements.txt


Run the notebook or Python script

📌 Limitations

Does not verify factual accuracy

Sensitive to dataset bias

May misclassify short or ambiguous news headlines

Deploy using Streamlit or Flask

Cross-dataset validation
