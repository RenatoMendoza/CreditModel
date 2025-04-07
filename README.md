# FIR Credit Model

The FIR (Financial Individual Risk) Credit Model is a traditional credit scoring tool built in Python to evaluate individual creditworthiness. Inspired by conventional frameworks like FICO, this model is designed to deliver interpretable credit scores and binary credit categories for automated lending decisions.

## 📌 Project Overview

This project helps financial institutions and small lenders assess credit risk using a transparent and explainable scoring system. The model outputs:

- A **credit score** scaled similarly to FICO (typically 400–850).
- A **binary classification**: Good or Bad credit risk.

## ⚙️ Key Features

- Based on key financial variables such as payment history and outstanding debt.
- Score weighting tailored through data-driven analysis.
- Evaluation tools include:
  - Score distribution histogram.
  - Confusion matrix.
  - Accuracy metrics.

## 🧮 Variable Selection

| Variable                | Description                                               | Role in Credit Behavior                                   |
|-------------------------|-----------------------------------------------------------|------------------------------------------------------------|
| avg_credit_history      | Average credit age (months)                               | Reflects stability and experience with credit              |
| avg_delay               | Average payment delay (days)                              | Indicates payment punctuality                              |
| avg_num_inquires        | Number of recent credit inquiries                         | Suggests recent credit-seeking activity                    |
| avg_outstanding_debt    | Total unpaid debt                                         | Measures financial burden                                  |
| avg_credit_mix          | Quality of credit mix (numerically mapped)                | Reflects diversity of credit usage                         |

## ⚖️ Weight Assignation

| Variable                | Weight (%) |
|-------------------------|------------|
| Payment History         | 35%        |
| Credit History Length   | 15%        |
| Credit Inquiries        | 15%        |
| Outstanding Debt        | 15%        |
| Credit Mix              | 20%        |

Weights are assigned based on domain knowledge and statistical impact, but can be adjusted for different institutional policies.

## 📊 Results Interpretation

### Score Distribution Histogram

This graph shows how credit scores are distributed among users.

![Score Distribution Histogram](sandbox:/mnt/data/image.png)

As seen above, most scores range between 400 and 850 with observable clusters around 650, 700, and 750. This suggests strong separation between user profiles and real-world realism in credit scoring.

### Confusion Matrix

The confusion matrix compares predicted vs actual user classifications.

![Confusion Matrix](sandbox:/mnt/data/image.png)

- **Good–Good:** 7060 (correctly predicted good credit)
- **Bad–Bad:** 2686 (correctly predicted bad credit)
- **Bad–Good:** 1244 (risky misclassifications)
- **Good–Bad:** 1097 (missed lending opportunities)

### 📈 Model Accuracy

**Overall Accuracy:** `0.8063`  
This means that over 80% of the classifications made by the FIR model are correct, demonstrating a reliable level of performance for production or institutional use.

## 🧠 How to Use

1. Load and clean your credit dataset.
2. Apply the variable selection and preprocessing functions.
3. Use the scoring function to generate a credit score.
4. Classify the score as 'good' or 'bad' using defined thresholds.
5. Evaluate using the included visualization tools.

## 📁 Folder Structure

