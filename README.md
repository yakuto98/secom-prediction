\# ⚡ Semiconductor Process Failure Prediction



> Predictive maintenance system for semiconductor fab using ML on the SECOM

> manufacturing dataset (UCI ML Repository)



🔗 \*\*Live Demo:\*\* https://secom-prediction-ycxvdkwkcrpftmwxukwkaf.streamlit.app/

📂 \*\*Repo:\*\* https://github.com/yakuto98/secom-prediction



\---



\## Business Problem



Unplanned wafer failures in semiconductor fabs are costly — every undetected

failure that reaches final test represents wasted process time, scrap

material, and lost yield. This project builds an early-warning system that

flags at-risk wafers based on in-line sensor readings, before they fail

final test.



The dataset (SECOM, UCI ML Repository) contains 590 sensor readings from

1,567 wafers collected during real semiconductor manufacturing, with a

binary pass/fail label per wafer.



\---



\## Results



| Metric | Score |

|---|---|

| Recall (Fail class) | 0.62 |

| Precision (Fail class) | 0.17 |

| F1 Score (Fail class) | 0.26 |

| AUC-ROC | 0.685 |

| Accuracy | 0.77 |



\*\*Confusion Matrix (test set, n=314):\*\*



| | Predicted Pass | Predicted Fail |

|---|---|---|

| \*\*Actual Pass\*\* | 228 | 65 |

| \*\*Actual Fail\*\* | 8 | 13 |



The model was deliberately optimized for \*\*recall over precision\*\*: in a

fab setting, the cost of an undetected failure (downtime, scrap, yield

loss) is significantly higher than the cost of a false alarm (one extra

inspection). The model correctly identifies 13 of 21 actual failures in

the test set.



\---



\## Key Technical Decisions



\*\*Class imbalance (93% pass / 7% fail).\*\*

Compared SMOTE oversampling vs. `class\_weight='balanced'`. SMOTE produced

inflated cross-validation scores (F1 ≈ 0.98) due to data leakage when

applied before cross-validation splitting — a mistake caught and corrected

by moving SMOTE inside an `imblearn.Pipeline` so it only sees training

folds. After fixing the leakage, true CV performance dropped to F1 ≈ 0.19.

`class\_weight='balanced'` ultimately outperformed SMOTE, likely because

SMOTE's synthetic samples are less reliable with only \~83 minority

examples in the training set.



\*\*High dimensionality (590 → 80 features).\*\*

Dropped sensors with >40% missing values and zero variance (590 → 296

features), then used `SelectKBest` (ANOVA F-test) to select the top 80

features most correlated with the failure label. Swept k from 20–100;

k=80 gave the best, most stable cross-validated F1.



\*\*Model selection.\*\*

Compared XGBoost, Random Forest, and Logistic Regression on the same

preprocessing pipeline. Logistic Regression with L2 regularization

(C=0.1) outperformed the tree-based models — with only \~83 failure

examples and up to 296 features, the tree ensembles overfit more readily

than a regularized linear model.



\*\*Missing data (\~4.5% of sensor readings).\*\*

Median imputation, chosen for robustness to the outliers common in

sensor data.



\---



\## Tech Stack



Python · scikit-learn · XGBoost · imbalanced-learn · Streamlit · Plotly



\---



\## Project Structure



```

secom-prediction/

├── data/                  # SECOM dataset (UCI ML Repository)

├── notebooks/

│   └── 01\_exploration.ipynb

├── src/

│   ├── preprocess.py

│   ├── train.py

│   └── evaluate.py

├── app/

│   └── dashboard.py        # Streamlit dashboard

├── models/

│   └── final\_model.pkl

├── results/

│   ├── 01\_data\_overview.png

│   └── metrics\_summary.txt

└── requirements.txt

```



\---



\## Running Locally



```bash

git clone https://github.com/yakuto98/secom-prediction.git

cd secom-prediction

python -m venv venv

venv\\Scripts\\activate          # Windows

pip install -r requirements.txt

streamlit run app/dashboard.py

```



\---



\## Dataset Source



McCann, M. \& Johnston, A. (2008). SECOM Dataset. UCI Machine Learning

Repository. https://archive.ics.uci.edu/dataset/179/secom

