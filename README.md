# Academic Performance Prediction Among Sri Lankan Private Campus Students

## A Comparative Study of Academic Performance Prediction Among Sri Lankan Private Campus Students Using Survey-Based Factors

This repository contains the data-analysis workflow, machine-learning implementation, evaluation results, figures, and supporting materials developed for our undergraduate research study at the Faculty of Information Technology, Horizon Campus, Sri Lanka.

The study examines whether academic, behavioural, and study-related information collected directly from students can support the classification of their current academic performance into three categories:

- Low
- Average
- High

Three machine-learning algorithms were compared:

- Logistic Regression
- Decision Tree
- Random Forest

The project focuses on predictive relationships found in the collected survey data. The results should not be interpreted as evidence that any individual factor directly causes academic performance.

---

# 1. Project Overview

Academic performance is influenced by many factors. These may include attendance, study habits, motivation, time management, sleep, stress, access to learning resources, and personal circumstances.

Many previous academic-performance prediction studies rely on institutional information such as GPA, examination marks, attendance databases, or Learning Management System records.

This study takes a different practical approach by using information that can be collected directly from students through a questionnaire.

The survey includes information related to:

- Class attendance
- Study hours
- LMS usage
- Assignment-submission behaviour
- Academic participation
- Sleep duration
- Motivation
- Time management
- Internet quality
- Part-time employment
- Academic stress
- Travel time
- Study-resource availability

The research focuses specifically on undergraduate students studying at private higher education institutions in Sri Lanka.

---

# 2. Research Aim

The aim of this study is to compare selected machine-learning models for predicting the academic performance level of Sri Lankan private campus undergraduate students using survey-based academic and behavioural factors.

---

# 3. Research Question

**To what extent can survey-based academic and behavioural factors support the prediction of Low, Average, and High academic performance among Sri Lankan private campus undergraduate students?**

---

# 4. Research Objectives

The study was conducted using the following objectives:

1. To collect survey-based academic and behavioural information from undergraduate students studying at private higher education institutions in Sri Lanka.

2. To preprocess and prepare the collected data for machine-learning analysis.

3. To develop Logistic Regression, Decision Tree, and Random Forest classification models.

4. To compare the performance of the three models using appropriate classification metrics.

5. To examine class-wise predictive performance, with particular attention to the Low academic-performance group.

6. To investigate which survey-based factors contribute most strongly to the predictions made by the models.

---

# 5. Data Collection

Primary data were collected using an anonymous Google Forms questionnaire.

The data-collection period was:

**28 June 2026 to 30 September 2026**

The questionnaire was distributed among undergraduate students studying at private higher education institutions in Sri Lanka.

The final downloaded survey dataset contained:

**435 responses**

The questionnaire included:

- Consent and eligibility questions
- Academic background information
- Study-related behaviour
- Academic and personal conditions
- Current academic-performance category
- Academic-support requirement

The raw participant-level dataset is not included in this public repository.

---

# 6. Participant Screening

Participant screening was carried out before model development.

The original dataset contained:

**435 responses**

After applying the eligibility requirements, the final modelling dataset contained:

**396 eligible participants**

Summary:

| Item | Count |
|---|---:|
| Raw survey responses | 435 |
| Final eligible participants | 396 |
| Excluded responses | 39 |

Eligibility checking considered:

- Informed consent
- Current student status
- Institution type
- Valid academic-performance target
- General data-quality requirements

The screening conditions were considered together. Therefore, individual screening counts may overlap and should not be treated as independent sequential exclusions.

---

# 7. Final Academic-Performance Distribution

The final eligible sample contained three academic-performance categories.

| Academic Performance Level | Number of Students | Percentage |
|---|---:|---:|
| Low | 110 | 27.78% |
| Average | 228 | 57.58% |
| High | 58 | 14.65% |
| **Total** | **396** | **100%** |

The Average category was the largest class, while the High category was the smallest.

Because the classes were not equally distributed, model performance was not evaluated using accuracy alone.

---

# 8. Target Variable

The target variable represents the student's current self-reported academic-performance level.

The three categories were defined as:

| Performance Level | Definition |
|---|---|
| Low | Below 40 marks |
| Average | 40–69 marks |
| High | 70 marks or above |

For modelling, the classes were encoded as:

```text
Low     = 0
Average = 1
High    = 2
```

The models therefore classify students' **current self-reported academic-performance category**.

The study does not claim to predict future examination results.

---

# 9. Predictor Variables

A total of **16 predictor variables** were included in the machine-learning models.

They were:

1. Year of study
2. Degree area / study field
3. Study mode
4. Average class attendance
5. Average study hours per day
6. LMS / online-learning-platform usage
7. Assignment-submission habits
8. Participation in lectures, tutorials, and practical sessions
9. Average sleep hours
10. Motivation level
11. Time-management ability
12. Internet-access quality
13. Part-time employment
14. Academic-stress level
15. Travel time to campus
16. Access to study resources

These variables were selected because they represent academic, behavioural, and study-related conditions that may be associated with student performance.

---

# 10. Variables Excluded from Model Training

Several survey fields were intentionally excluded from the predictor set.

These included:

- Consent information
- Eligibility-screening questions
- Gender
- Current CA / coursework marks
- Academic-support requirement
- Academic-performance target

Gender was used only for descriptive analysis and was not used as a machine-learning predictor.

The current CA / coursework marks variable was also excluded.

CA marks are closely related to the academic-performance target. Including them could introduce **target leakage**, making the model appear more accurate than it would be when using independent survey-based factors.

---

# 11. Data Preparation

The original imported dataset was preserved during the analysis, and a separate working copy was used for processing.

The preparation process included:

1. Loading the survey dataset
2. Checking the number of rows and columns
3. Reviewing data types
4. Checking missing values
5. Reviewing possible duplicate records
6. Examining response categories
7. Applying eligibility criteria
8. Selecting the approved predictor variables
9. Separating predictors and target
10. Standardizing category labels
11. Preparing the variables for machine-learning pipelines

The original raw dataset was not overwritten.

---

# 12. Data Preprocessing

Different variable types required different preprocessing methods.

## 12.1 Nominal Variables

Nominal variables were handled using one-hot encoding.

Examples include:

- Degree area
- Study mode

---

## 12.2 Ordinal Variables

Ordered survey categories were mapped according to their natural order.

Examples include:

- Attendance
- Study hours
- LMS usage
- Assignment-submission behaviour
- Academic participation
- Sleep
- Motivation
- Time management
- Internet quality
- Academic stress
- Travel time
- Study-resource access

---

## 12.3 Missing Values

Missing predictor values were handled using most-frequent-value imputation.

Imputation was performed inside the machine-learning pipeline.

---

## 12.4 Feature Scaling

StandardScaler was used for Logistic Regression.

Decision Tree and Random Forest did not require scaling.

All learned preprocessing operations were kept inside scikit-learn pipelines to reduce information leakage between training and validation data.

---

# 13. Machine-Learning Models

Three supervised classification algorithms were developed and compared.

---

## 13.1 Logistic Regression

Logistic Regression was used as a simple and interpretable baseline model.

The model used:

- L2 regularization
- Balanced class weights
- Maximum iterations = 1000

The following values of `C` were tested:

```text
0.1
1
10
```

---

## 13.2 Decision Tree

Decision Tree was included because it can model non-linear relationships and provides an interpretable tree structure.

The tuning grid included:

```text
max_depth:
3
5
8
None
```

and:

```text
min_samples_leaf:
1
3
5
```

Balanced class weights were used.

---

## 13.3 Random Forest

Random Forest combines multiple decision trees and can model more complex relationships between the predictor variables and the target.

The tuning grid included:

```text
n_estimators:
100
200
```

```text
max_depth:
5
10
None
```

```text
min_samples_leaf:
1
2
4
```

Balanced class weights were also used.

---

# 14. Random Seed

A fixed random state was used to improve reproducibility.

```text
random_state = 42
```

---

# 15. Validation Strategy

The final model comparison used **nested cross-validation**.

The outer evaluation used:

**Stratified 5-Fold Cross-Validation**

The inner hyperparameter search used:

**Stratified 3-Fold GridSearchCV**

The hyperparameter search optimized:

**Macro F1-score**

The purpose of nested cross-validation was to separate hyperparameter selection from final model-performance estimation.

The three models were compared using the same outer folds.

---

# 16. Evaluation Metrics

The following metrics were used:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Class-wise Recall
- Confusion Matrix
- Mean performance across folds
- Standard deviation across folds

---

## Primary Evaluation Metric

The main evaluation metric was:

**Macro F1-score**

Macro F1 gives equal importance to the Low, Average, and High classes.

This was particularly useful because the target classes were imbalanced.

---

# 17. Baseline Model Results

Baseline models were evaluated before final hyperparameter tuning.

| Model | Accuracy Mean | Macro Precision Mean | Macro Recall Mean | Macro F1 Mean |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6087 | 0.5774 | 0.6357 | 0.5856 |
| Decision Tree | 0.5885 | 0.5105 | 0.5103 | 0.5080 |
| Random Forest | **0.6668** | **0.6493** | 0.5522 | 0.5732 |

These results provided a reference point for evaluating the effect of hyperparameter tuning.

---

# 18. Baseline vs Tuned Models

Hyperparameter tuning improved Macro F1 for all three algorithms.

| Model | Baseline Macro F1 | Tuned Macro F1 | Improvement |
|---|---:|---:|---:|
| Logistic Regression | 0.5856 | 0.5962 | +0.0106 |
| Decision Tree | 0.5080 | 0.5160 | +0.0080 |
| Random Forest | 0.5732 | 0.6145 | +0.0413 |

The largest improvement was observed for Random Forest.

---

# 19. Final Model Performance

The final nested cross-validation results were:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6188 | 0.5877 | **0.6513** | 0.5962 |
| Decision Tree | 0.5429 | 0.5134 | 0.5628 | 0.5160 |
| Random Forest | **0.6668** | **0.6267** | 0.6190 | **0.6145** |

Random Forest achieved the highest mean Accuracy, Macro Precision, and Macro F1.

However, Logistic Regression achieved the highest Macro Recall.

This indicates that no single model was strongest according to every evaluation measure.

---

# 20. Model Stability

Variation in Macro F1 across the five outer folds was also examined.

| Model | Mean Macro F1 | SD | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.5962 | **0.0466** | 0.5441 | 0.6601 |
| Decision Tree | 0.5160 | 0.0539 | 0.4538 | 0.5664 |
| Random Forest | **0.6145** | 0.0739 | 0.5159 | 0.7099 |

Logistic Regression showed the smallest variation across the outer folds.

Random Forest achieved the highest average Macro F1 but showed greater fold-to-fold variation.

---

# 21. Class-Wise Recall

Class-wise recall was examined because overall metrics can hide differences between target classes.

| Model | Low Recall | Average Recall | High Recall |
|---|---:|---:|---:|
| Logistic Regression | **0.7273** | 0.5524 | **0.6742** |
| Decision Tree | 0.6364 | 0.4959 | 0.5561 |
| Random Forest | 0.5727 | **0.7448** | 0.5394 |

Logistic Regression achieved the strongest recall for the Low academic-performance group.

Random Forest achieved the strongest recall for the Average group.

This is an important finding because model selection depends on the intended purpose of the system.

---

# 22. Interpretation of Model Performance

The study did not identify one model that was best for every purpose.

### Random Forest performed best for:

- Accuracy
- Macro Precision
- Macro F1

### Logistic Regression performed best for:

- Macro Recall
- Low-class recall
- High-class recall
- Fold-to-fold stability

### Decision Tree

Decision Tree produced lower overall predictive performance compared with Logistic Regression and Random Forest.

Therefore, the most suitable model depends on the practical objective.

If the main goal is stronger overall multiclass classification, Random Forest is the stronger model in this dataset.

If the main goal is to identify a larger proportion of students in the Low academic-performance category, Logistic Regression may be more useful.

---

# 23. Statistical Comparison

Fold-level Macro F1 scores were compared using the Friedman test.

The result was:

```text
Friedman statistic = 4.80
p-value = 0.0907
α = 0.05
```

Because:

```text
p > 0.05
```

the analysis did not find a statistically significant difference among the three models at the 5% significance level.

According to the predefined analysis plan, pairwise Wilcoxon signed-rank tests were therefore not performed.

This statistical result should be interpreted carefully because it is based on five outer cross-validation folds from one dataset.

---

# 24. Model Interpretation

Several methods were used to understand the trained models.

These included:

- Logistic Regression coefficients
- Decision Tree visualization
- Random Forest feature importance
- Permutation importance

These approaches were used to better understand how different survey-based variables contributed to the model predictions.

The interpretation is predictive rather than causal.

---

# 25. Important Predictive Factors

Permutation importance identified the following variables among the strongest predictive factors in the fitted Random Forest model.

| Rank | Predictor | Mean Importance |
|---:|---|---:|
| 1 | Access to study resources | 0.0787 |
| 2 | Travel time | 0.0494 |
| 3 | Degree area | 0.0490 |
| 4 | Part-time employment | 0.0446 |
| 5 | LMS usage | 0.0390 |
| 6 | Study hours | 0.0221 |
| 7 | Attendance | 0.0216 |
| 8 | Year of study | 0.0189 |
| 9 | Motivation | 0.0147 |
| 10 | Academic stress | 0.0147 |

These values indicate predictive importance within the fitted model.

They do not prove that these variables cause changes in academic performance.

---

# 26. Research Figures

The final research figures are stored in:

```text
results/figures/
```

The repository contains the following figures:

```text
baseline_model_comparison.png
baseline_vs_tuned.png
class_wise_recall.png
decision_tree_structure.png
permutation_importance.png
random_forest_feature_importance.png
target_distribution.png
tuned_decision_tree_confusion_matrix.png
tuned_logistic_regression_confusion_matrix.png
tuned_random_forest_confusion_matrix.png
```

These figures provide visual summaries of the model-development and evaluation process.

---

# 27. Result Tables

The aggregated research tables are stored in:

```text
results/tables/
```

The final result files include:

```text
accuracy_95CI.csv
baseline_model_results.csv
baseline_vs_tuned.csv
feature_importance_comparison.csv
final_class_recall_table.csv
final_model_results.csv
final_model_stability.csv
final_participant_screening_flow.csv
final_sample_summary.csv
final_target_distribution.csv
final_top_predictive_factors.csv
logistic_regression_coefficients.csv
macro_f1_95CI.csv
nested_cv_fold_results.csv
permutation_importance.csv
statistical_summary.csv
tuned_overfitting_check.csv
```

These files contain aggregated analytical results and do not contain raw participant questionnaire responses.

---

# 28. Final Google Colab Notebook

The final analysis notebook is stored in:

```text
notebooks/Academic_Performance_Final_Research.ipynb
```

The notebook contains the main analytical workflow used for the study.

This includes:

- Dataset inspection
- Data-quality checking
- Eligibility screening
- Predictor selection
- Data preprocessing
- Baseline model development
- Hyperparameter tuning
- Nested cross-validation
- Performance evaluation
- Class-wise recall
- Confusion matrices
- Statistical testing
- Model interpretation
- Feature importance
- Final result generation

The public version of the notebook is shared without saved raw participant-level outputs.

---

# 29. Repository Structure

The main repository structure is:

```text
academic-performance-prediction-private-campus-student/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── diagrams/
│
├── docs/
│
├── notebooks/
│   ├── README.md
│   └── Academic_Performance_Final_Research.ipynb
│
├── results/
│   ├── README.md
│   │
│   ├── figures/
│   │   ├── README.md
│   │   └── research figures
│   │
│   └── tables/
│       ├── README.md
│       └── aggregated result tables
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 30. Source Code

The repository contains separate Python scripts supporting the project.

## `src/preprocessing.py`

This script supports:

- Loading the survey data
- Cleaning variables
- Checking data quality
- Applying eligibility rules
- Selecting predictors
- Preparing the target
- Preparing the modelling dataset

---

## `src/train_models.py`

This script supports:

- Preprocessing pipelines
- Logistic Regression
- Decision Tree
- Random Forest
- Cross-validation
- Hyperparameter tuning
- Model training

---

## `src/evaluate_models.py`

This script supports:

- Performance evaluation
- Class-wise recall
- Confusion matrices
- Model comparison
- Statistical analysis
- Evaluation outputs

The final Colab notebook contains the most complete version of the final analytical workflow.

---

# 31. Technologies Used

The project was developed using:

- Python
- pandas
- NumPy
- scikit-learn
- SciPy
- Matplotlib
- Google Colab
- Git
- GitHub

---

# 32. Installation

Clone the repository:

```bash
git clone https://github.com/sandeepaseshan2-commits/academic-performance-prediction-sri-lanka.git
```

Open the project directory:

```bash
cd academic-performance-prediction-sri-lanka/academic-performance-prediction-private-campus-student
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# 33. Reproducibility

The project was developed with reproducibility in mind.

Important analytical settings are documented in the repository, including:

- Predictor definitions
- Target definition
- Variable exclusions
- Category mappings
- Preprocessing methods
- Model settings
- Hyperparameter grids
- Cross-validation strategy
- Evaluation metrics
- Statistical testing
- Random seed

The main random state used in the study was:

```text
42
```

The final Google Colab notebook provides the main record of the analysis used to generate the reported results.

Because the participant-level dataset is private, complete reproduction of the original analysis requires authorized access to the survey dataset or another dataset with the same structure.

---

# 34. Data Privacy

Participant privacy was considered throughout the project.

The public repository does not contain:

- Raw Google Form responses
- Raw Excel survey data
- Raw CSV survey data
- Cleaned participant-level datasets
- Individual student records
- Participant-level predictions
- Personally identifying information

Only the following materials are made public:

- Analysis code
- Research documentation
- Aggregated result tables
- Research figures
- Model-evaluation outputs
- Clean public Colab notebook

This allows the analytical process to remain transparent without publicly exposing individual survey responses.

---

# 35. Data Availability

The raw participant-level survey data are not publicly available because the responses were collected from students under privacy conditions.

Aggregated analytical results, figures, documentation, and analysis code are available through this repository.

Any future sharing of participant-level data would need to follow appropriate institutional and ethical requirements.

---

# 36. Code Availability

The Python analysis code and final Google Colab notebook are available in this repository.

The available code covers:

- Eligibility screening
- Data preprocessing
- Model development
- Hyperparameter tuning
- Nested cross-validation
- Model evaluation
- Statistical testing
- Feature importance
- Result generation

---

# 37. Study Limitations

Several limitations should be considered when interpreting the findings.

First, the survey used voluntary participation and convenience-based recruitment. The final sample therefore should not automatically be considered representative of every private-campus undergraduate student in Sri Lanka.

Second, the predictors and academic-performance category were self-reported by participants.

Third, the three target classes were not equally distributed.

Fourth, the study used nested internal cross-validation but did not use an independent external validation dataset.

Fifth, the study is cross-sectional. The predictors and academic-performance category were collected during the same general period.

Therefore, the results represent predictive relationships within the collected dataset and should not be interpreted as causal evidence.

The models should also not be used by themselves to make high-stakes academic decisions about individual students.

---

# 38. Practical Interpretation

The results suggest that survey-based academic and behavioural information contains useful predictive signals related to students' current academic-performance categories.

The study also shows that model choice depends on the intended purpose.

Random Forest produced the strongest overall Macro F1 and accuracy.

However, Logistic Regression identified a larger proportion of students in the Low-performance category.

This difference is important if machine learning is considered for future academic-support applications.

A prediction model should be used only as one source of information together with academic advice, student communication, and professional judgement.

---

# 39. Future Work

Future research could improve and extend this study by:

- Increasing the sample size
- Collecting data from a wider range of private higher education institutions
- Using an independent external validation dataset
- Collecting longitudinal student data
- Comparing additional machine-learning models
- Studying model calibration
- Examining subgroup performance
- Using additional explainability methods
- Combining survey data with institutional academic records
- Developing an academic-support prototype
- Evaluating the model in a real educational setting

---

# 40. Project Status

The following major stages have been completed:

- Research questionnaire preparation
- Survey-data collection
- Final dataset preparation
- Data-quality checking
- Participant eligibility screening
- Predictor selection
- Target definition
- Target-leakage prevention
- Data preprocessing
- Logistic Regression development
- Decision Tree development
- Random Forest development
- Baseline model evaluation
- Hyperparameter tuning
- Nested cross-validation
- Accuracy evaluation
- Macro Precision evaluation
- Macro Recall evaluation
- Macro F1 evaluation
- Class-wise recall analysis
- Confusion-matrix generation
- Statistical comparison
- Model-stability analysis
- Logistic Regression interpretation
- Decision Tree visualization
- Random Forest feature importance
- Permutation importance
- Final research figure generation
- Final result-table generation
- Final Google Colab analysis
- GitHub documentation

The repository now contains the main computational work used to support preparation of the final research manuscript.

---

# 41. Authors

## Wathsala Kithulgala

**Index Number:** ITBIN-2312-0025  
**Faculty:** Faculty of Information Technology  
**Institution:** Horizon Campus, Sri Lanka

Main areas of contribution included:

- Research development
- Questionnaire and data collection
- Machine-learning development
- Model interpretation
- Research documentation
- Manuscript preparation

---

## W. Seshan Sandeepa

**Index Number:** ITBIN-2312-0024  
**Faculty:** Faculty of Information Technology  
**Institution:** Horizon Campus, Sri Lanka

Main areas of contribution included:

- GitHub repository management
- Data preparation
- Machine-learning workflow development
- Model evaluation
- Statistical analysis
- Research documentation
- Manuscript preparation

---

# 42. Research Title

**A Comparative Study of Academic Performance Prediction Among Sri Lankan Private Campus Students Using Survey-Based Factors**

---

# 43. Academic and Ethical Notice

This repository was developed as part of an undergraduate academic research project.

The machine-learning models are intended for research and educational purposes.

They should not independently be used to determine:

- Student grades
- Academic penalties
- Admissions
- Scholarships
- Disciplinary actions
- Other high-stakes educational decisions

Machine-learning predictions should support, rather than replace, academic judgement and direct engagement with students.

---

# 44. Citation

If the final research paper is published, the complete journal citation will be added here.

Until then, the research may be referred to using the project title:

> Wathsala Kithulgala and W. Seshan Sandeepa,  
> **“A Comparative Study of Academic Performance Prediction Among Sri Lankan Private Campus Students Using Survey-Based Factors.”**

---

# 45. Repository Purpose

The main purpose of this repository is to provide a transparent record of the computational work completed for the study.

It allows readers to understand:

- How the survey data were prepared
- How participant eligibility was determined
- Which predictors were used
- Which variables were excluded
- How target leakage was reduced
- How the machine-learning models were developed
- How hyperparameters were selected
- How model performance was evaluated
- How the models were statistically compared
- How model interpretation was carried out
- How the final research conclusions were supported

The repository acts as supporting material for the final research paper while keeping participant-level survey information private.
