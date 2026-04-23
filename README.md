# Telco Customer Churn Prediction

## Problem
Telecom companies lose significant revenue when customers cancel their subscriptions.
This project predicts which customers are likely to churn so the company can intervene
early — offering discounts or better plans before losing them.

---

## Dataset
- **Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 real customer records
- **Features:** 20 features including contract type, tenure, monthly charges, and internet service type
- **Target:** Churn (Yes/No)

---

## Project Structure
```
telco-churn/
│
├── notebook.ipynb       # Full analysis — EDA, preprocessing, models
├── README.md            # Project overview
└── dataset/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## What I Did

### 1. Exploratory Data Analysis (EDA)
- Analysed churn distribution — found **27% churn rate** (class imbalance)
- Key finding: **month-to-month contract customers churn significantly more** than annual customers
- Fiber optic internet customers churn at higher rates despite faster speeds — likely due to higher cost
- Senior citizens churn at higher rates than non-senior customers
- Customers with higher monthly charges are more likely to churn

### 2. Data Preprocessing
- Fixed `TotalCharges` column stored as text — converted to float
- Dropped `customerID` — irrelevant for prediction
- Applied **Label Encoding** for binary columns (gender, Partner, Dependents, etc.)
- Applied **One Hot Encoding** for nominal columns (MultipleLines, InternetService, Contract, PaymentMethod)
- Applied **SMOTE** to handle class imbalance — balanced training data from 73/27 to 50/50
- Applied **StandardScaler** for feature scaling

### 3. Models Tested
Tested 5 machine learning models with SMOTE applied to training data:

| Model | Accuracy | Churn Recall | Churn F1 |
|---|---|---|---|
| Logistic Regression | 76% | **83%** | 65% |
| Decision Tree | 73% | 55% | 52% |
| SVM | 78% | 74% | 64% |
| Random Forest | 79% | 60% | 60% |
| ANN (MLP) | 75% | 63% | 57% |

---

## Results

**Best Model: Logistic Regression + SMOTE**

- Achieved **83% recall on churners** — catches 83% of customers about to leave
- Overall accuracy: 76%

Logistic Regression outperformed more complex models like Random Forest and ANN.
This confirms that for small, structured tabular datasets, simpler models often
perform better than complex ones.

---

## Key Findings

1. **Contract type is the strongest churn predictor** — month-to-month customers churn far more than annual or two-year contract customers
2. **Tenure matters** — new customers are much more likely to churn than long-term ones
3. **High monthly charges increase churn risk** — customers paying more are more likely to leave
4. **ANN performed worst** — confirming neural networks are better suited for unstructured data like images and text, not tabular data

---

## Business Recommendation

Target month-to-month contract customers who have been with the company less than
12 months and pay above-average monthly charges. These are the highest churn risk
customers and should be offered loyalty discounts or contract upgrade incentives.

---

## Tech Stack
- Python 3.13
- pandas, NumPy
- Seaborn, Matplotlib
- scikit-learn
- imbalanced-learn (SMOTE)

---

## What I Learned
- Real-world data has hidden issues — `TotalCharges` appeared clean but had blank strings masking null values
- Class imbalance significantly affects model performance — SMOTE improved churn recall from 60% to 83%
- Recall is more important than accuracy for churn prediction — missing a churner costs the business money
- Simpler models can outperform complex ones on structured tabular data

---

## Author
**Shaik Mohammed Umar Ayaan**
- GitHub: https://github.com/umarayaan/telco-customer-churn
- Email: umarshaik355@gmail.com

---

*Note: In future iterations I will apply SMOTE before scaling for best practice.*
