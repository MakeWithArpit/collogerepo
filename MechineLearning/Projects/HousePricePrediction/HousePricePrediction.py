import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data.csv').copy()
data.drop(columns=["street", "country"], inplace=True)

data["date"] = pd.to_datetime(data["date"])
data["sale_year"] = data["date"].dt.year
data.drop(columns=["date"], inplace=True)

data = data[data["price"] < 3000000]
data = data[data["sqft_living"] < 8000]

data = pd.get_dummies(data, columns=["city", "statezip"], drop_first=True)

X = data.drop("price", axis=1)
y = data["price"]


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()