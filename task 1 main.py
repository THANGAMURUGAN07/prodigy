import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('train.csv')  # Replace 'your_dataset.csv' with the actual file name

# Select features and target variable
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
y = data['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()

# Plot residuals
residuals = y_test - predictions
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()

# You can now use the trained model to predict house prices for new data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate mean squared error
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

# Calculate mean absolute error
mae = mean_absolute_error(y_test, predictions)
print('Mean Absolute Error:', mae)

# Calculate R-squared
r2 = r2_score(y_test, predictions)
print('R-squared:', r2)
