import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('f:/housing.csv')
h_df = df.copy()

# Ensure correct data types
h_df[['floors', 'bathrooms', 'bedrooms']] = h_df[['floors', 'bathrooms', 'bedrooms']].astype(int)

# Function to replace outliers with NaN using IQR
def replace_outliers_with_nan_iqr(df, feature, inplace=False): 
    desired_feature = df[feature]
    q1, q3 = desired_feature.quantile([0.25, 0.75]) 
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr 
    lower_bound = q1 - 1.5 * iqr
    indices = (desired_feature[(desired_feature > upper_bound) | (desired_feature < lower_bound)]).index
    if inplace:
        df.loc[indices, feature] = np.nan
    else:
        return desired_feature.replace(desired_feature[indices].values, np.nan)

# Detect and list numerical features for outlier handling
numeric_features = h_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('price')  # Avoid touching target
features_with_outlier = []

# Detect outliers first
for col in numeric_features:
    q1, q3 = h_df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr 
    lower_bound = q1 - 1.5 * iqr
    if h_df[(h_df[col] > upper_bound) | (h_df[col] < lower_bound)].shape[0] > 0:
        features_with_outlier.append(col)

# Replace outliers with NaN
for col in features_with_outlier:
    replace_outliers_with_nan_iqr(h_df, col, inplace=True)

# Drop unused columns
h_df.drop(['street', 'date', 'country'], axis=1, inplace=True)

# Encode categorical variables
h_df['statezip'], _ = pd.factorize(h_df['statezip'])

# Fill missing values (optional: use median or mean)
h_df.fillna(h_df.median(numeric_only=True), inplace=True)

# Split features and target
X = h_df.drop('price', axis=1)
y = h_df['price']

# Compute Mutual Information
def get_mi_score(X, y):
    mi = mutual_info_regression(X, y, random_state=10)
    mi = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return mi

# Drop non-numeric columns if still present
X = X.select_dtypes(include=[np.number])

mi_score = get_mi_score(X, y)

# Plot MI scores
plt.figure(figsize=(12, 5))
ax = sns.barplot(y=mi_score.index[1:], x=mi_score[1:])
ax.set_title('MI scores', fontdict={'fontsize': 16})
plt.show()

# Drop least informative feature (based on MI or other logic)
X.drop(['yr_renovated'], axis=1, inplace=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Train model
model = RandomForestRegressor(n_estimators=50, random_state=10)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)



mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)



print('MSE: ', mse)
print('RMSE: ', rmse)
