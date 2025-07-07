# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.impute import SimpleImputer
# import pickle

# # Load training data
# df = pd.read_csv("train.csv")
# X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
# y = df['SalePrice']

# # Impute missing values
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X)

# # Train model
# model = LinearRegression()
# model.fit(X_imputed, y)

# # Save model and imputer
# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

# with open("imputer.pkl", "wb") as f:
#     pickle.dump(imputer, f)

# print("✅ Model and imputer saved!")


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import pickle

# Load dataset
df = pd.read_csv("train.csv")

# Select relevant features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'GarageArea', 'YearBuilt']
target = 'SalePrice'

df = df[features + [target]].dropna()

# Separate features and target
X = df[features]
y = df[target]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train linear regression model
model = LinearRegression()
model.fit(X_imputed, y)

# Save the model and imputer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

print("✅ Model and imputer saved with updated features!")
