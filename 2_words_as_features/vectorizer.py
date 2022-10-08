##AUTHOR: Rose Rutherford-Stone (27062516)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import Perceptron
import gzip
import json
import collections

#EXTRACT DATA FROM ZIP FOLDER
jsonfiledirectory = "/Users/rosiers/Documents/GitHub/472-assignment1/goemotions.json.gz"

with gzip.open(jsonfiledirectory, "r") as f:
    data = json.loads(f.read().decode("utf-8"))

#2.1
strings = []
sentiment = []
for item in data:
    strings.append(item[0])
    sentiment.append(item[2])

vectorizer = CountVectorizer()
# tokenize and build vocab
v_fit = vectorizer.fit_transform(strings)

#2.2
X_train, X_test, y_train, y_test = train_test_split(v_fit, sentiment, test_size=0.2)

#2.3
#Base-MNP ~ 2.3.1
base_mnb = MultinomialNB()
base_mnb.fit(X_train,y_train)
print("Base-MNP: ")
print(base_mnb.score(X_test,y_test))

#Base-DT ~ 2.3.2
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
print("Base-DT: ")
print(dtc.score(X_test, y_test))

#Base-MLP ~ 2.3.3
p = Perceptron()
p.fit(X_train,y_train)
print("Base-MLP: ")
print(p.score(X_test,y_test))

#Top-MNB ~ 2.3.4


#Top-DT ~ 2.3.5


#Top-MLP ~ 2.3.6