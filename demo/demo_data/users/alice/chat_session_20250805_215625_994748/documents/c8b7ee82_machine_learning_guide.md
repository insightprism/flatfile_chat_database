# Machine Learning Guide

## Data Preprocessing
Data preprocessing is a crucial step in machine learning that involves cleaning and transforming raw data into a format suitable for modeling.

### Handling Missing Values
- **Numerical data**: Use mean, median, or mode imputation
- **Categorical data**: Use most frequent category or create a separate 'missing' category
- **Advanced methods**: KNN imputation, iterative imputation

### Feature Scaling
Feature scaling ensures all features contribute equally to the model:
- **StandardScaler**: Scales features to have mean=0 and std=1
- **MinMaxScaler**: Scales features to a fixed range (usually 0-1)
- **RobustScaler**: Uses median and IQR, robust to outliers

### Encoding Categorical Variables
- **One-hot encoding**: Creates binary columns for each category
- **Label encoding**: Assigns numerical values to categories
- **Target encoding**: Uses target variable statistics
