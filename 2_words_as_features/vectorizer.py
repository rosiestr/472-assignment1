##AUTHOR: Rose Rutherford-Stone (27062516)
import gzip
import json
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.linear_model import Perceptron
<<<<<<< Updated upstream
import gzip
import json
import collections
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
=======
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
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
=======
nb = MultinomialNB()

param_grid = {
    'alpha' : [0.0,0.5,1.0,2.0 ]
}
nb_grid = GridSearchCV(nb,param_grid)
nb_grid.fit(X_train, y_train)
print("TOP-MNB: ")
print(nb_grid.score(X_test, y_test))


# In[37]:
>>>>>>> Stashed changes


#Top-DT ~ 2.3.5
tdt = DecisionTreeClassifier()

<<<<<<< Updated upstream
#will have to check what these parameters do and how to comment on them 

=======
>>>>>>> Stashed changes
params_dt = {
    "criterion" : ['entropy'],
    "max_depth" : [2, 10],
    "min_samples_split" : [1.0, 5, 100]
}
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
dt_grid = GridSearchCV(
    estimator=tdt,
    param_grid= params_dt
)
<<<<<<< Updated upstream

dt_grid = dt_grid.fit(X_train, y_train)
print("TOP-DT: ")
print(dt_grid.score(X_test, y_test))

#Top-MLP ~ 2.3.6
=======
dt_grid = dt_grid.fit(X_train, y_train)
print("TOP-DT: ")
print(dt_grid.score(X_test, y_test))


# In[ ]:


#Top-MLP ~ 2.3.6
mlp = MLPClassifier(max_iter=1)

param_grid = {
    'activation' : ['sigmoid', 'tanh', 'relu', 'identity'],
    'hidden_layer_sizes': [(10,10,10), (30,50)],
    'solver': ['sgd', 'adam']
}
mlp_grid = GridSearchCV(mlp,param_grid,n_jobs=-1)
mlp_grid.fit(X_train, y_train)
print("TOP-MLP: ")
print(mlp_grid.score(X_test, y_test))


#Confusion matrix_BASE-MNB

#make predictions on test data
y_pred = base_mnb.predict(X_test)

#y_test=actual, y_pred=prediction
cm_mnb = confusion_matrix(y_test,y_pred)
print(cm_mnb)

#for plotting the matrix grid
color = 'blue'
matrix = plot_confusion_matrix(base_mnb, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()

print(classification_report(y_test, y_pred))



#Confusion marix_BASE-DT
y_pred = dtc.predict(X_test)

cm_dt = confusion_matrix(y_test, y_pred)
print(cm_dt)

print(classification_report(y_test, y_pred))



#Confusion matrix_BASE-MLP
y_pred = p.predict(X_test)

cm_p = confusion_matrix(y_test, y_pred)
print(cm_p)

print(classification_report(y_test, y_pred))

#Confusion matrix_TOP-MNB
y_pred = nb_grid.predict(X_test)

cm_Tmnb = confusion_matrix(y_test, y_pred)
print(cm_Tmnb)

print(classification_report(y_test, y_pred))


#Confusion matrix_TOP-DT
y_pred = dt_grid.predict(X_test)

cm_Tdt = confusion_matrix(y_test, y_pred)
print(cm_Tdt)

print(classification_report(y_test, y_pred))


#Confusion matrix_TOP-MLP
y_pred = mlp_grid.predict(X_test)

#cm_mlp = confusion_matrix(y_test, y_pred)
#print(cm_mlp)

print(classification_report(y_test, y_pred))
>>>>>>> Stashed changes
