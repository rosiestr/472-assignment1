##AUTHOR: Krishna Patel 40176352 
import gzip
import json
import matplotlib.pyplot as plt
import numpy as np

#EXTRACT DATA FROM ZIP FOLDER
jsonfiledirectory = "C:\\Users\\Krish\\.vscode\\472-assignment1\\goemotions.json.gz"

with gzip.open(jsonfiledirectory, "r") as f:
    data = json.loads(f.read().decode("utf-8"))

print(len(data))
# emotionSet = set()
# sentimentList = ['Positive', 'Negative', 'Ambiguous', 'Neutral']

# for x in data:
#     while len(emotionSet)<= 28:
#         emotionSet.add(x[1])
#         break
    
#     if len(emotionSet)==28 :
#         break

# emotionList = list(emotionSet)
# print(emotionList)
# print(sentimentList)
# print("test")

#COUNT EMOTIONS AND CREATE PIE CHART
emotionCounter = {}

for x in data: 
    emotionCounter[x[1]] = emotionCounter.get(x[1], 0)+1

emotionList = emotionCounter.keys()
emotionValues = emotionCounter.values()

plt.pie(emotionValues, labels = emotionList, autopct='%1.2f%%')
plt.show() 

#COUNT SENTIMENTS AND CREATE PIE CHART
sentimentCounter= {}
for x in data: 
    sentimentCounter[x[2]] = sentimentCounter.get(x[2], 0)+1

sentimentList = sentimentCounter.keys()
sentimentValues = sentimentCounter.values()

plt.pie(sentimentValues, labels=sentimentList, autopct='%1.2f%%')
plt.show()

#%%
