# Classification Models for Breast Cancer Diagnosis

## Project Overview
This project aimed to build a predictive model that can accurately classify breast cancer diagnosis as either benign or malignant based on the provided features. Logistic regression was used as the primary modeling technique, and different methods of handling missing data were explored to improve the model's performance.

## Missing Data
The dataset contained a total of 186 missing variables (~5% of the dataset). Six features were identified to contain 31 missing data points each, including ‘radius_mean’, ‘texture_mean’, ‘radius_se’, ‘texture_se’, ‘radius_worst’, and texture_worst’. 

**Handling Missing Data**

Missing data was handled with the aim of minimising its impact on the models while maintaining the dataset integrity and ensuring prediction validity. The following methods were used:

- Complete Case Analysis (CCA)
- Mean Imputation using MICE
- Hot Deck Imputation using MICE
- Linear Regression Prediction-Based Imputation using MICE

## Model Development
Logistic regression models were developed using different regularisation methods (L1-Lasso, L1+L2-Elastic Net, L2-Ridge) and the preprocessed datasets. The models was trained using cross-validation to select the best regularisation parameter.

## Evaluation Reults
- In breast cancer classification, sensitivity and specificity are more crucial than overall accuracy. Maximising sensitivity ensures that most true malignant cases are correctly identified, while maximising specificity reduces false alarms and unnecessary treatments.
- Based on the evaluation metrics, classifiers with **Hot Deck Imputation** performed best across all regularisation techniques, particularly in terms of sensitivity and specificity. This suggests that Hot Deck Imputation effectively handled missing data while preserving the original data distribution. 
- Among the regularisation techniques, **Elastic Net regularisation** optimised the classifier with Hot Deck Imputation the most, likely due to its combined approach of Lasso and Ridge regularisation, which minimises overfitting while retaining key features.

## Limitations
- While the models showed high performance on the current dataset, their generalisability may be limited. Future studies focussed on external validation using independent datasets will be needed to ensure the models' generalisability to new data and real patients. 
- Training the models on larger datasets would improve their robustness and ability to capture complex patterns without overfitting. The small dataset size (n=569) and the presence of missing values (186) were limitations that increased the risk of overfitting.
