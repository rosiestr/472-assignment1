import gzip
import json

jsonfiledirectory = "C:\\Users\\Krish\\.vscode\\472-assignment1\\goemotions.json.gz"

with gzip.open(jsonfiledirectory, "r") as f:
    data = json.loads(f.read().decode("utf-8"))

print(data[3])

print(type(data[3]))
y=data[3]
print(y[0])
print(y[1])
print(y[2])

emotionSet = set()
sentimentList = ['Positive', 'Negative', 'Ambiguous', 'Neutral']

for x in data:
    while len(emotionSet)<= 28:
        emotionSet.add(x[1])
        break
    
    if len(emotionSet)==28 :
        break

emotionList = list(emotionSet)
print(emotionList)
print(sentimentList)
print("test")

print(data.count(emotionList[1]))

