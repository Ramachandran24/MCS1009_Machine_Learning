import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

fake = pd.read_csv('f:/fake.csv')
true = pd.read_csv('f:/true.csv')


fake['sentiment'] = 0
true['sentiment'] = 1


dataset = pd.concat([true, fake], ignore_index=True)


dataset.drop(['date', 'subject'], axis=1, inplace=True)


input_array = np.array(dataset['title'])

corpus = []
ps = PorterStemmer()

for i in range(min(40000, len(input_array))):  # Ensure index range is valid
    review = re.sub('[^a-zA-Z]', ' ', str(input_array[i]))
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    corpus.append(' '.join(review))


cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:len(corpus), dataset.columns.get_loc("sentiment")].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


classifier = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2)


classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

#Output
#Confusion Matrix:
# [[0 1]
# [0 0]]