### Chapter Summaries

### Chapter 1: DataFrames, Feature Engineering, and Exploratory Tools

This chapter introduced foundational skills in data manipulation using **pandas**. The focus was on understanding how to:

- Load and explore tabular data with `.head()`, `.info()`, and `.describe()`
- Access and modify data using `.loc[]`, `.iloc[]`, boolean masking, and method chaining
- Perform **column-wise transformations** using `.map()`, `.apply()`, and conditional logic
- Clean and engineer features manually, including:
  - Mapping categorical values to numerical codes
  - Creating new features from existing ones
  - Handling missing or inconsistent data

The exercises used the **Titanic dataset** to explore survival patterns and prepare features for downstream machine learning.

---

### Chapter 2: Custom Transformers and Data Pipelines

This chapter focused on building reusable, clean preprocessing logic using **scikit-learn pipelines**. Key topics included:

- Creating **custom transformers** by extending `BaseEstimator` and `TransformerMixin`
- Writing transformers for:
  - `CustomMappingTransformer`: mapping text categories to numeric codes
  - `CustomOHETransformer`: one-hot encoding a single column using `pd.get_dummies`
  - `CustomDropColumnsTransformer`: dropping or keeping specified columns with error/warning handling
- Assembling a full **Pipeline** to automate data preprocessing for a customer ratings dataset
- Steps included: dropping IDs, mapping gender and experience level, and one-hot encoding OS and ISP fields

By the end of the chapter, we constructed a robust and modular pipeline that can be reused and extended for model training and evaluation.

---

### Chapter 3: GitHub Libraries and Correlation-Based Feature Selection

This chapter focused on organizing reusable code and enhancing feature engineering through correlation analysis.

- Introduced the use of a **GitHub-hosted `library.py`** file to store and reuse custom transformers across notebooks
- Practiced importing library code using `wget` and `%run` in Jupyter/Colab
- Explored **Pearson correlation** to identify multicollinearity between features
- Applied **NumPy masking** (`np.triu`, `np.fill_diagonal`) to isolate the upper triangle of a correlation matrix
- Built `CustomPearsonTransformer`:
  - Calculates correlation in `fit()` and identifies highly correlated columns
  - Drops correlated columns in `transform()` with an error check if `fit()` hasn't been called

This chapter emphasized **modular design**, **code reuse**, and **data quality improvements** in preprocessing workflows.

---

### Chapter 4: Outlier Handling and Clipping with Custom Transformers

This chapter focused on managing outliers using statistical rules and expanding the transformer library with new tools.

- Introduced the **3-sigma rule** for detecting extreme values
- Created `CustomSigma3Transformer` to clip a column within ±3 standard deviations of the mean
- Created `CustomTukeyTransformer` to clip values using Tukey’s **inner** or **outer fences** based on IQR
- Both transformers were added to `library.py`, followed scikit-learn conventions, and included clear assertion checks
- Compared `.query()` and `.drop()` for row filtering and noted their handling of NaNs

By the end, we had reusable tools for clipping outliers and a deeper understanding of how different statistical thresholds affect the data.

---

### Chapter 5: Scaling, Skewness Reduction, and Robust Pipelines

This chapter focused on scaling data and reducing skewness for more effective preprocessing.

- Applied **Box-Cox transformation** to the `Fare` column:
  - Tested lambdas from -5 to 5 to minimize skewness
  - Skipped NaNs and selected the best lambda based on skew
- Built **CustomRobustTransformer** to scale specific columns using **median** and **IQR**:
  - Skipped binary columns with zero IQR
  - Added assertion checks for fit status and valid columns
- Created a wrapped version using **sklearn's RobustScaler**
- Applied these tools to the **Cable Customer dataset**:
  - Pipeline included dropping columns, mapping categories, one-hot encoding, imputing missing values, and scaling **'Time Spent'**
  - Scaled **'Age'** for consistent feature ranges

By the end, we had robust tools for handling skewed and outlier-prone data, integrated into flexible pipelines.

---

### Chapter 6: Distance Metrics and KNN Imputation

This chapter explored similarity measures and practical imputation using K-nearest neighbors.

- Calculated **Euclidean distance** between feature vectors using list comprehensions
- Computed **cosine similarity** to compare the orientation of two vectors, demonstrating how small angle differences affect similarity scores
- Built **CustomKNNTransformer**:
  - Wrapped `KNNImputer` from scikit-learn
  - Ensured pandas DataFrame input/output
  - Added assertion checks for fit status and valid input types
- Integrated **CustomKNNTransformer** into a full **pipeline**:
  - Dropped IDs, mapped categorical fields, and one-hot encoded OS and ISP
  - Scaled **'Time Spent'** and **'Age'** using **CustomRobustTransformer**
  - Applied KNN imputation after scaling for consistent, complete data

By the end, we had practical tools for distance-based imputation and a refined pipeline capable of handling missing values in real-world datasets.
