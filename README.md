What is Credit Card Fraud Detection?

Credit card fraud detection is a classical problem in the field of Machine Learning Classification.
The primary objective is to build a predictive model that can ingest transaction details (like amount,
time, and hidden feature embeddings) and instantly classify the transaction into one of two
categories: Genuine (0) or Fraudulent (1).

The Fundamental Challenge: Class Imbalance
In the real world, fraudulent transactions make up a tiny fraction of total transactions (often
less than 0.2%). This severe class imbalance means a model could achieve 99.8% accuracy
simply by guessing "Genuine" every time. However, such a model would be entirely useless
because it would miss every single fraud case. Therefore, handling imbalance is the central
theme of this project.

How Do We Measure Success?
Because overall "Accuracy" is a misleading metric here, we evaluate our models using alternative
metrics:

Metric Meaning in Fraud Detection Goal

Precision

Out of all transactions the model
flagged as fraud, how many were
actually fraud?

High precision means fewer false
alarms (annoyed customers whose
cards get incorrectly blocked).

Recall
(Sensitivity)

Out of all the actual fraudulent
transactions, how many did the model
catch?

High recall means the bank loses
less money to fraudsters. This is
usually prioritized.

F1-Score

The harmonic mean of Precision and
Recall.

Provides a balanced view when you
want to avoid both false alarms and
missed frauds.

AUC-ROC

Area Under the Receiver Operating
Characteristic Curve. Measures how
well the model separates the two
classes.

A score of 1.0 is perfect; 0.5 is
random guessing. Above 0.90 is
excellent.


Project Workflow
Data Loading & Structure: Reads the dataset and checks for missing values.

Outlier Detection: Analyzes transaction amounts. Note: Outliers are deliberately kept as they often represent fraudulent activities.

Data Preprocessing:
Converts integer columns to numeric to ensure compatibility with the ROSE package.
Scales the Amount and Time features using standard scaling.
Encodes the Class target variable as a categorical factor (Genuine vs Fraud).

Exploratory Data Analysis (EDA):
Visualizes the severe class imbalance.
Plots feature distributions (Genuine vs. Fraud) to identify highly discriminative variables.
Generates a correlation heatmap.

Train-Test Split: Splits the data (70% training, 30% testing) to evaluate model generalization.
Handling Class Imbalance: Applies the ROSE algorithm to the training set to create a synthetic, balanced distribution of Genuine and Fraud cases.

Modeling: Trains three distinct models:

Logistic Regression: A baseline statistical classification approach.

Random Forest: An ensemble method that provides high accuracy and handles non-linear relationships well.

Decision Tree (rpart): A highly interpretable tree-based model mapped out visually.

Evaluation: Compares models using Confusion Matrices, capturing Precision, Recall, F1-Score, and plots ROC curves to determine the best-performing algorithm.

How to Run
Place creditcard.csv in the same directory as the R script.

Run the script sequentially in RStudio or your preferred R environment.

Review the console outputs and generated plots (in the Plots pane) for insights and model performance comparison.

Evaluation Metrics Focus
In fraud detection, standard accuracy is extremely misleading. This project focuses on:

Recall (Sensitivity): Crucial for catching as many frauds as possible (minimizing bank losses).

Precision: Important for minimizing false alarms (preventing genuine customer cards from being blocked).

AUC-ROC: A comprehensive measure of the model's ability to distinguish between the two classes.
