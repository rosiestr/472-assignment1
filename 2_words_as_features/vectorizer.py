##AUTHOR: Rose Rutherford-Stone (27062516)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import Perceptron
import gzip
import json
import collections
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 

#EXTRACT DATA FROM ZIP FOLDER
jsonfiledirectory = "C:\\Users\\Krish\\.vscode\\472-assignment1\\goemotions.json.gz"

with gzip.open(jsonfiledirectory, "r") as f:
    data = json.loads(f.read().decode("utf-8"))

#2.1
strings = []
sentiment = []
emotions =[]
for item in data:
    strings.append(item[0])
    sentiment.append(item[2])
    emotions.append(item[1])

vectorizer = CountVectorizer()
# tokenize and build vocab
v_fit = vectorizer.fit_transform(strings)


#2.2
#X_train, X_test, y_train, y_test = train_test_split(v_fit, sentiment, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(v_fit, emotions, test_size=0.2)

#2.3
#Base-MNB ~ 2.3.1
base_mnb = MultinomialNB()
base_mnb.fit(X_train,y_train)
print("Base-MNP: ")
print(base_mnb.score(X_test,y_test))

#Base-DT ~ 2.3.2
# dtc = DecisionTreeClassifier()
# dtc.fit(X_train, y_train)
# print("Base-DT: ")
# print(dtc.score(X_test, y_test))

#Base-MLP ~ 2.3.3
# p = Perceptron()
# p.fit(X_train,y_train)
# print("Base-MLP: ")
# print(p.score(X_test,y_test))

#Top-MNB ~ 2.3.4
tmnb = MultinomialNB()
params = {
    "alpha" : [0.5, 0.0, 10.0, 100.0]
}
grid_search = GridSearchCV(
    estimator= tmnb,
    param_grid= params
)

grid_search = grid_search.fit(X_train, y_train)
print("TOP_MNB:")
print(grid_search.score(X_test, y_test))


#Top-DT ~ 2.3.5
tdt = DecisionTreeClassifier()

#will have to check what these parameters do and how to comment on them 

params_dt = {
    "criterion" : ['entropy'],
    "max_depth" : [2, 10],
    "min_samples_split" : [1.0, 5, 100]
}

dt_grid = GridSearchCV(
    estimator=tdt,
    param_grid= params_dt
)

dt_grid = dt_grid.fit(X_train, y_train)
print("TOP-DT: ")
print(dt_grid.score(X_test, y_test))

#Top-MLP ~ 2.3.6
