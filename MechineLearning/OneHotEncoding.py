from sklearn.preprocessing import OneHotEncoder
city = [["Bareilly"], ["Rampur"], ["jaipur"]]
encoder = OneHotEncoder()
result = encoder.fit_transform(city)
print(result.toarray())
