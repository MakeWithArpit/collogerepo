import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "StudyHours":[1,2,3,4,5],
    "Marks":[55,45,10,70,20]
}

df = pd.DataFrame(data)
X = df[['StudyHours']]
y = df['Marks']

model = LinearRegression()
model.fit(X,y)
prediction = model.predict([[8]])
print("Marks of 8 hours study:",prediction[0])

plt.scatter(X,y)
plt.plot(X,model.predict(X))
plt.xlabel("StudyHours")
plt.ylabel("Marks")
plt.show()
