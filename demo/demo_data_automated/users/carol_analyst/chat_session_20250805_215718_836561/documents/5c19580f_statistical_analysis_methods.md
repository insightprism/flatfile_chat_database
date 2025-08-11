# Statistical Analysis Methods for Data Science

## Introduction
This comprehensive guide covers statistical methods commonly used in data science, from basic descriptive statistics to advanced modeling techniques.

## Descriptive Statistics

### Central Tendency
- **Mean**: Best for symmetric distributions
- **Median**: Robust to outliers, good for skewed data
- **Mode**: Useful for categorical data

### Variability Measures
- **Standard Deviation**: Shows spread in same units as data
- **Variance**: Useful for mathematical operations
- **Interquartile Range (IQR)**: Robust measure of spread
- **Coefficient of Variation**: Allows comparison across different scales

### Distribution Shape
- **Skewness**: Measures asymmetry
- **Kurtosis**: Measures tail heaviness
- **Normality Tests**: Shapiro-Wilk, Kolmogorov-Smirnov

## Inferential Statistics

### Hypothesis Testing Framework
1. **State Hypotheses**: H₀ (null) and H₁ (alternative)
2. **Choose Significance Level**: Typically α = 0.05
3. **Select Test Statistic**: Based on data type and assumptions
4. **Calculate p-value**: Probability of observing data given H₀ is true
5. **Make Decision**: Reject H₀ if p-value < α

### Common Statistical Tests

#### Parametric Tests
```python
# t-test for comparing means
from scipy import stats

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(sample_data, population_mean)

# Two-sample t-test (independent)
t_stat, p_value = stats.ttest_ind(group1, group2)

# Paired t-test (dependent)
t_stat, p_value = stats.ttest_rel(before, after)

# ANOVA for comparing multiple groups
f_stat, p_value = stats.f_oneway(group1, group2, group3)
```

#### Non-Parametric Tests
```python
# Mann-Whitney U test (non-parametric alternative to t-test)
u_stat, p_value = stats.mannwhitneyu(group1, group2)

# Wilcoxon signed-rank test (paired alternative)
w_stat, p_value = stats.wilcoxon(before, after)

# Kruskal-Wallis test (alternative to ANOVA)
h_stat, p_value = stats.kruskal(group1, group2, group3)
```

## Regression Analysis

### Linear Regression
**Assumptions**:
1. Linearity
2. Independence
3. Homoscedasticity (constant variance)
4. Normality of residuals

**Model Diagnostics**:
```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Fit model
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()

# Diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs fitted
axes[0,0].scatter(model.fittedvalues, model.resid)
axes[0,0].set_title('Residuals vs Fitted')

# Q-Q plot for normality
sm.qqplot(model.resid, line='s', ax=axes[0,1])

# Scale-Location plot
axes[1,0].scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)))
axes[1,0].set_title('Scale-Location')

# Residuals vs leverage
axes[1,1].scatter(model.get_influence().hat_matrix_diag, model.resid)
axes[1,1].set_title('Residuals vs Leverage')
```

### Logistic Regression
For binary outcomes:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Fit model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predictions
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]

# Evaluation
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.3f}")
```

## Time Series Analysis

### Components of Time Series
1. **Trend**: Long-term movement
2. **Seasonality**: Regular patterns
3. **Cyclicality**: Irregular long-term patterns
4. **Noise**: Random variation

### Decomposition
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
decomposition = seasonal_decompose(ts_data, model='additive')
decomposition.plot()
```

### Stationarity Testing
```python
from statsmodels.tsa.stattools import adfuller

# Augmented Dickey-Fuller test
result = adfuller(ts_data)
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
```

### ARIMA Modeling
```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(ts_data, order=(1,1,1))
fitted_model = model.fit()

# Forecasting
forecast = fitted_model.forecast(steps=12)
```

## Survival Analysis

### Kaplan-Meier Estimation
```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed)

# Plot survival curve
kmf.plot_survival_function()
```

### Cox Proportional Hazards Model
```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df, duration_col='T', event_col='E')
cph.print_summary()
```

## Multivariate Analysis

### Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

### Cluster Analysis
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Evaluate clustering
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg:.3f}")
```

## Experimental Design

### A/B Testing
```python
# Power analysis for sample size calculation
from statsmodels.stats.power import ttest_power

# Calculate required sample size
power = ttest_power(effect_size=0.2, nobs=None, alpha=0.05, power=0.8)
print(f"Required sample size per group: {power:.0f}")

# Statistical test for A/B test results
control_conversion = 0.10
treatment_conversion = 0.12
control_n = 1000
treatment_n = 1000

# Chi-square test for proportions
from scipy.stats import chi2_contingency

contingency_table = np.array([
    [control_conversion * control_n, (1 - control_conversion) * control_n],
    [treatment_conversion * treatment_n, (1 - treatment_conversion) * treatment_n]
])

chi2, p_value, dof, expected = chi2_contingency(contingency_table)
```

### Factorial Design
For studying multiple factors simultaneously:
- **2^k Design**: k factors, each at 2 levels
- **Latin Square**: Controls for two sources of variation
- **Split-Plot Design**: Different randomization for different factors

## Bayesian Analysis

### Bayesian Framework
P(H|D) = P(D|H) × P(H) / P(D)

Where:
- P(H|D): Posterior probability
- P(D|H): Likelihood
- P(H): Prior probability
- P(D): Evidence

### Example: Bayesian A/B Testing
```python
import pymc3 as pm

with pm.Model() as model:
    # Priors
    p_A = pm.Beta('p_A', alpha=1, beta=1)
    p_B = pm.Beta('p_B', alpha=1, beta=1)
    
    # Likelihood
    obs_A = pm.Binomial('obs_A', n=n_A, p=p_A, observed=successes_A)
    obs_B = pm.Binomial('obs_B', n=n_B, p=p_B, observed=successes_B)
    
    # Derived quantity
    lift = pm.Deterministic('lift', (p_B - p_A) / p_A)
    
    # Sample
    trace = pm.sample(2000, tune=1000)
```

## Model Validation and Selection

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# k-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Stratified k-fold for imbalanced data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf)
```

### Information Criteria
- **AIC (Akaike Information Criterion)**: Penalizes model complexity
- **BIC (Bayesian Information Criterion)**: Stronger penalty for complexity
- **AICc**: Corrected AIC for small samples

```python
# Calculate AIC for linear regression
n = len(y)
mse = np.mean((y - y_pred)**2)
k = X.shape[1]  # number of parameters
aic = n * np.log(mse) + 2 * k
```

## Statistical Assumptions and Violations

### Common Violations and Solutions

#### Multicollinearity
- **Detection**: Variance Inflation Factor (VIF)
- **Solutions**: Remove correlated predictors, ridge regression, PCA

#### Heteroscedasticity
- **Detection**: Breusch-Pagan test, White test
- **Solutions**: Weighted least squares, robust standard errors

#### Non-normality
- **Detection**: Shapiro-Wilk test, Q-Q plots
- **Solutions**: Transformations, non-parametric tests

#### Autocorrelation
- **Detection**: Durbin-Watson test, ACF plots
- **Solutions**: Add lagged variables, time series models

## Best Practices

1. **Always Visualize First**: Plots reveal patterns statistics might miss
2. **Check Assumptions**: Don't blindly apply statistical tests
3. **Use Appropriate Sample Sizes**: Power analysis is crucial
4. **Multiple Comparisons**: Adjust p-values when testing multiple hypotheses
5. **Effect Size Matters**: Statistical significance ≠ practical significance
6. **Report Confidence Intervals**: More informative than just p-values
7. **Reproducible Research**: Set random seeds, document methodology
8. **Domain Knowledge**: Statistics should complement, not replace, subject expertise

## Advanced Topics

### Machine Learning vs. Traditional Statistics
- **ML Focus**: Prediction accuracy
- **Statistics Focus**: Understanding relationships and uncertainty
- **Hybrid Approaches**: Statistical learning, interpretable ML

### Causal Inference
- **Randomized Experiments**: Gold standard for causality
- **Observational Studies**: Requires careful design
- **Methods**: Instrumental variables, regression discontinuity, propensity score matching

### Big Data Considerations
- **Sampling**: When full data analysis isn't feasible
- **Multiple Testing**: More hypotheses = higher false discovery rate
- **Computational Statistics**: Bootstrap, permutation tests, MCMC

This guide provides a foundation for statistical analysis in data science. Remember that statistical methods are tools - the key is choosing the right tool for your specific problem and data characteristics.
