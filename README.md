# HIV PREDICTION MODEL

![alt text](tmp/download.png)

## Table of Contents

- [Project Overview](#overview)
- [Business Understanding](#business-understanding) 
- [Data Understanding](#data-understanding)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
- [Modeling](#modeling)  
- [Key Takeaways](#key-takeaways-from-our-analysis)  
- [Instructions](#Instructions)
- [Recommendations](#recommendations)  
- [Future Improvements](#future-improvements)  
- [Tech Stack & Dependencies](#tech-stack--dependencies)  


# Overview

HIV remains a major global health issue, affecting approximately 38 million people worldwide, with adolescent girls and young women (AGYW) at higher risk due to biological, socio-economic, and behavioral factors. Sub-Saharan Africa bears nearly 70% of global cases, with Kenya facing significant challenges among AGYW due to gender disparities, poverty, and limited access to education and healthcare. Despite aid-funded programs, gaps in reaching high-risk groups persist. This project will apply the CRISP-DM methodology to enhance data-driven HIV intervention strategies in the health sector.

# Business Understanding
## Problem Statement
Adolescent girls and young women in Kenya face a disproportionately high risk of HIV infection. Despite health program interventions which target structural drivers such as poverty, gender inequality, and lack of education—critical challenges persist:
  - Inefficient Resource Allocation: Limited ability to identify and prioritize high-risk populations.
  - Gaps in Intervention Impact: Uncertainty about which strategies (biomedical, behavioral, or social) are most effective.
  - Data-Driven Decision-Making: Need for actionable insights to optimize program reach and reduce HIV incidence.

This project addresses these challenges by leveraging data science to predict risk, evaluate interventions, and guide evidence-based policy decisions.

### Key Stakeholders
|Stakeholder	            |      Interest                                        |
|---------------------------|------------------------------------------------------|
|Public Health Agencies     | Reduce HIV incidence, measure program ROI.           |
|NGOs & Program Coordinators| Improve intervention targeting and scalability.      |
|Policymakers               | Design policies backed by data-driven insights.      |
|Healthcare Providers       | Identify at-risk individuals for early intervention. |



### **Objectives:**
1. Predict HIV Risk: Develop a machine learning model to estimate HIV risk based on demographics, behavior, and program exposure.
2. Identify Risk Factors: Use feature importance techniques to highlight contributors to HIV vulnerability.
3. Optimize Resource Allocation: Provide actionable insights to improve program targeting and service delivery.

# Data Understanding
## 1. Dataset Overview
- Source: SPARK (Strengthening Prevention and Awareness for Resilient Communities) program data (2018–2022).
- Size: 455,807 records × 42 features (pre-cleaning).

***Key Variables:***
  - Demographics: Age, county, household structure, parental status.
  - Socioeconomic: Household size, income sources, food security.
  - Behavioral: School attendance, sexual activity, condom use.
  - Medical: HIV testing history, test results (result: 99.8% negative, 0.2% positive).
  - Interventions: Participation in biomedical (e.g., HTS testing), behavioral, and social programs.

## 2. Data Quality & Cleaning
**Key Issues Identified:**
  1. Missing Values:
- High missingness (>80%) in columns like dreams_program_other and exit_reason_other (dropped).
- Moderate missingness in critical features: county (2%), currently_in_school (0.9%), ever_had_sex (1.7%).
- Resolution:
  - Numerical columns imputed with median (e.g., age_of_household_head).
  - Categorical columns filled with mode (e.g., county → "Nairobi").
 2. Class Imbalance:
   - Severe imbalance in target variable (result): 99.8% HIV-negative, 0.2% HIV-positive.
   - Instead of using SMOTE (Synthetic Minority Oversampling) for balancing the classes during modeling, we applied undersampling.
 3. Outliers:
   - Extreme values in age_of_household_head (max = 727 million due to data entry errors).
   - Resolution: Replaced outliers with median values.


# Exploratory Data Analysis (EDA)
- Demographics: 72% of participants lived in households headed by parents.

 ![alt text](tmp/6fdeae7b-33d4-4b9b-b255-24638c8c3a4f.png)

- Behavioral Trends:
42% used condoms inconsistently with their last partner.

![alt text](tmp/e6e2aa64-3022-4d13-9bb7-4f86f5d27218.png)

- Impact of the health program: Graduation emerges as the frontrunner exit reason at an impressive 398,182, overshadowing the next largest categories—lost to follow-up.

![alt text](tmp/34b374fe-f151-440d-ba19-98bac8134562.png)

- Currently in school

![alt text](tmp/31d23d5b-fbdf-4739-9082-f0eecefbc704.png)

# Modeling 
## 1. Approach

*Objective:* To build a classification model to predict HIV risk (binary: Positive/Negative) and identify key drivers of infection.

*Framework:* We employed the CRISP-DM methodology for a structured and iterative data mining process.

 **Model Selection**
- Logistic Regression: Baseline model for interpretability and benchmarking.
- Random Forest: Handles non-linear relationships and feature interactions; robust to outliers.
- XGBOOST: Gradient-boosted tree model optimized for accuracy and speed; handles class imbalance effectively.

  **class Imbalance Mitigation**
SMOTE (Synthetic Minority Oversampling): Generated synthetic HIV-positive samples to balance the target variable (0.2% → 50% representation).

## 2. Preprocessing Pipeline

**Feature Engineering:**
- Derived age from date_of_birth and date_of_enrollment.
- Binned age_at_first_sexual_encounter into categories (<15, 15–17, 18+).

**Encoding:**
- One-hot encoded categorical variables (e.g., county, current_income_source).
- Scaling: Standardized numerical features (e.g., household_size) for Logistic Regression.

## 3. Model Training & Tuning
**Data Splitting**

Train/Test Split: 80% training, 20% testing (stratified to preserve class balance).

**Hyperparameter Tuning**
- GridSearchCV: Optimized parameters across 5-fold cross-validation.
- Logistic Regression: Penalty (l1, l2), regularization strength (C).
- Random Forest: n_estimators (100–500), max_depth (5–20), min_samples_split (2–10).
- XGBoost: learning_rate (0.01–0.2), max_depth (3–10), subsample (0.6–1.0).

## 4. Evaluation Metrics

|Metric	    |Importance                                                         |
|-----------|-------------------------------------------------------------------|
|Recall	    |Prioritize identifying true positives (critical for HIV prevention)|
|Precision	|Avoid overloading resources with false positives.                  | 
|F1-Score	|Balance precision and recall for imbalanced data.                  |

## 5. Results
**Model Performance**

|Model	              |Accuracy	|Precision	|Recall	|F1-Score  |Key Tuning Parameters|
|---------------------|---------|-----------|-------|----------|----------------------|
|Logistic Regression  |69.61%	  |68.88%	    |68.18%	|68.53%	   |C=10, max_iter=100, solver='liblinear'
|Random Forest	      |69.85%	  |69.23%	    |68.18%	|68.70%	   |max_depth=20, min_samples_leaf=1, n_estimators=200
|XGBoost	            |66.67%   |66.32%	    |63.64% |64.95%	   |scale_pos_weight=50, learning_rate=0.01, max_depth=3, subsample=0.7

# Key Takeaways from Our Analysis
**Dealing with Class Imbalance**

Our initial models struggled due to extreme class imbalance, defaulting to trivial "Negative" predictions. 

**What Worked?**

Undersampling Helped: Once we balanced the data, recall and F1-score improved significantly.

Logistic Regression Stood Out: Among the baseline models, it performed the best on the balanced dataset.

**Tuning Results**

Random Forest - Best Overall: Achieved a solid balance between precision and recall as indicated by  F1-score (68.70%).

The confusion matrix below illustrates the performance of the tuned Random Forest model. It correctly predicted 150 HIV-negative and 135 HIV-positive cases, but misclassified 60 false positives and 63 false negatives. While overall accuracy is strong, the false negative rate suggests room for improvement to reduce missed high-risk cases. 

![alt text](tmp/a97b4165-520a-46b7-abe2-140bff311b8e.png)



### Key Features Identified (Random Forest Insights)
- Age at First Sexual Encounter: Early onset correlates with higher risk.
- Household Food Security: Food insecurity increases risk by 2.1×.
- Education Level: Lower educational attainment linked with a 1.8× increase in risk.
- Condom Use: Inconsistent use reflects a 1.5× higher risk scenario.

## Instructions
Clone this repository:

bash

Copy code

git clone <repository-url>

Run index.ipynb in Jupyter Notebook or any other compatible IDE.


# Recommendations
- Targeted Outreach: Deploy mobile testing units in high-risk counties like Nairobi and Kisumu.

- School-Based Initiatives: Fuse sexual health education with conditional cash transfers to boost engagement and reduce risk.
- Household-Level Support: Launch community kitchen programs to combat food insecurity head-on.
- Partner Engagement: Enhance male partner testing protocols in regions with significant age disparities.

# Future Improvements

Feature Engineering: Test interactions (e.g., age + behavioral factors) or explore temporal trends.

Better Data: Collect more positive cases or integrate external datasets to improve class balance.

Model Enhancements: Experiment with Neural Networks and Incorporate spatial analysis for geo-targeting.

## Tech Stack & Dependencies
Python 3.8+

Libraries: pandas, scikit-learn, xgboost, numpy, matplotlib

Models Employed:
- Logistic Regression
- Random Forest
- XGBOOST

## Resources
[Download Processed Data](https://github.com/RichieRicky/DSF-PT08P5-GROUP5_CAPSTONE-PROJECT/raw/refs/heads/main/dreams_raw_dataset.zip)

[Open Notebook](https://github.com/RichieRicky/DSF-PT08P5-GROUP5_CAPSTONE-PROJECT/blob/main/Index.ipynb)
