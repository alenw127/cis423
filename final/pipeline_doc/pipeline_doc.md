# Titanic Data Pipeline Documentation

## Pipeline Overview
This pipeline preprocesses the Titanic dataset to prepare it for machine learning modeling. It handles categorical encoding, target encoding, outlier detection and treatment, feature scaling, and missing value imputation.

<img src="https://github.com/alenw127/cis423/blob/main/final/screenshot/pipeline.PNG" width="70%" alt="Pipeline Diagram">

## Step-by-Step Design Choices

### 1. ID Column Dropping ('drop_id')
- **Transformer:** `CustomDropColumnsTransformer(['id'])`
- **Design Choice:** Remove unnecessary unique identifier
- **Rationale:** IDs are not informative for prediction and could introduce noise.

### 2. Ever Married Mapping (`map_married`)
- **Transformer:** `CustomMappingTransformer('ever_married', {'Yes': 1, 'No': 0})`
- **Design Choice:** Binary encoding of marital status
- **Rationale:** Simple and intuitive mapping without increasing feature dimensionality.

### 3. Residence Type Mapping (`map_residence`)
- **Transformer:** `CustomMappingTransformer('Residence_type', {'Urban': 0, 'Rural': 1})`
- **Design Choice:** Binary encoding of residence type
- **Rationale:** Allows the model to capture rural vs. urban differences efficiently.

### 4. Target Encoding for Gender (`target_gender`)
- **Transformer:** `CustomTargetTransformer(col='gender', smoothing=10)`
- **Design Choice:** Target encoding with smoothing factor
- **Rationale:** Converts categorical gender feature to a numeric representation based on its relationship with the target variable (stroke). Smoothing reduces overfitting from rare categories.

### 5. Target Encoding for Work Type (`target_work_type`)
- **Transformer:** `CustomTargetTransformer(col='work_type', smoothing=10)`
- **Design Choice:** Target encoding for work type
- **Rationale:** Work type may have multiple categories and rare levels; target encoding with smoothing balances between category means and the global mean.

### 6. Target Encoding for Smoking Status (`target_smoking`)
- **Transformer:** `CustomTargetTransformer(col='smoking_status', smoothing=10)`
- **Design Choice:** Target encoding for smoking status
- **Rationale:** Allows the model to capture nuanced relationships between smoking status and stroke risk while controlling for overfitting.

### 7. Outlier Treatment for Age (`tukey_age`)
- **Transformer:** `CustomTukeyTransformer(target_column='age', fence='outer')`
- **Design Choice:** Tukey method with outer fence
- **Rationale:** Removes extreme age outliers that could skew scaling and imputation.

### 8. Outlier Treatment for BMI (`tukey_bmi`)
- **Transformer:** `CustomTukeyTransformer(target_column='bmi', fence='outer')`
- **Design Choice:** Tukey method with outer fence
- **Rationale:** Handles extreme BMI outliers without discarding reasonable but high or low values.

### 9. Outlier Treatment for Average Glucose Level (`tukey_glucose`)
- **Transformer:** `CustomTukeyTransformer(target_column='avg_glucose_level', fence='outer')`
- **Design Choice:** Tukey method with outer fence
- **Rationale:** Controls the influence of extremely high glucose readings that could otherwise dominate the model.

### 10. Scaling Age (`scale_age`)
- **Transformer:** `CustomRobustTransformer('age')`
- **Design Choice:** Robust scaling
- **Rationale:** Consistent with age, ensures BMI is on a comparable scale and reduces outlier impact.

### 11. Scaling BMI (`scale_bmi`)
- **Transformer:** `CustomRobustTransformer('bmi')`
- **Design Choice:** Robust scaling
- **Rationale:** Handles extreme BMI outliers without discarding reasonable but high or low values.

### 12. Scaling Average Glucose Level (`scale_glucose`)
- **Transformer:** `CustomRobustTransformer('avg_glucose_level')`
- **Design Choice:** Robust scaling
- **Rationale:** Ensures that glucose values are scaled consistently with other numerical features.

### 13. KNN Imputation (`impute`)
- **Transformer:** `CustomKNNTransformer(n_neighbors=5)`
- **Design Choice:** KNN imputation using 5 nearest neighbors
- **Rationale:** Captures relationships between features and provides context-aware imputation, avoiding the oversimplification of mean/median imputation.

## Pipeline Execution Order Rationale
1. Drop ID column to eliminate non-informative identifiers.
2. Binary mappings for straightforward categorical features (ever married, residence type).
3. Target encodings for multi-category features that benefit from capturing the relationship with the target variable.
4. Outlier treatments applied before scaling to avoid outliers influencing scaling parameters.
5. Scaling ensures features are comparable and appropriate for distance-based imputation.
6. KNN imputation uses the fully preprocessed data to fill missing values with contextual accuracy.

## Random State

The value used in train_test_split is: 

rs = 121
