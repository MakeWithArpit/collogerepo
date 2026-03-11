from sklearn.cluster import KMeans

a = [
    [10],
    [20],
    [30],
    [40],
    [50],
    [60],
    [70],
    [80],
]


model = KMeans(n_clusters=2)
model.fit(a)
print(model.labels_)
