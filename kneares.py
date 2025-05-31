from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

X = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [8, 6]]
y = [0, 0, 0, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
