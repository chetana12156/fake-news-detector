
# Fake News Detection 

This project implements a machine learning model that classifies news articles as **FAKE** or **REAL** using natural language processing techniques.

## Overview

The goal of this project is to build a text classification system capable of identifying misinformation in news content. The model is trained on a labeled dataset of news articles and evaluated using standard performance metrics.

## Approach

The pipeline includes:

- Text preprocessing and feature extraction using **TF-IDF Vectorization**
- Model training using a **Passive Aggressive Classifier**
- Dataset splitting into training and testing sets
- Performance evaluation using accuracy and a confusion matrix

## Results

The model achieves approximately **93% accuracy** on the test dataset.

Confusion Matrix: [[591 47]
                  [ 42 587]]


The classifier performs consistently across both classes, showing balanced predictions for FAKE and REAL news.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn

## How to Run

1. Clone the repository.
2. Install the required dependencies: pip install pandas numpy scikit-learn
3. Run the script: python fake_news.py
                

The program will train the model, display performance metrics, and allow the user to input custom headlines for prediction.

## Future Improvements

- Implement additional classification models for comparison
- Add a web-based interface for user interaction
- Deploy the model as an online application
