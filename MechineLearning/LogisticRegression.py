from sklearn.linear_model import LogisticRegression

X = [[10],[20],[30],[80]]
Y = [0,0,1,0]

model = LogisticRegression()
model.fit(X,Y)

print(model.predict([[40]]))
