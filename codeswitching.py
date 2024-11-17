import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

file = '/Users/Sidhanth/Desktop/CodeSwitchNER/cleaned_train.csv'
chunks = []
header = pd.read_csv(file, nrows=0).columns.tolist()
for start in range(0, 500, 4800):
    chunk = pd.read_csv(file, skiprows=range(1, start + 1), nrows=500, names=header)
    chunks.append(chunk)
data = {
    'word': [],
    'label': []
}
vectorizers = []
classifiers = []
for chunk in chunks:
    for words, labels in zip(chunk['words_list'], chunk['labels']):
        wordList = words.split()
        labelList = labels.split()
        for word, label in zip(wordList, labelList):
            cleanWord = word.strip("[]'")
            cleanLabel = label.strip("[]'")
            data['word'].append(cleanWord)
            data['label'].append(cleanLabel)
    processed = pd.DataFrame(data)
    X = processed['word']
    y = processed['label']
    vectorizer = CountVectorizer(analyzer='word')
    XVectorized = vectorizer.fit_transform(X)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(XVectorized, y)
    vectorizers.append(vectorizer)
    classifiers.append(classifier)

def predlanglabel(sentence):
    labelMapping = {'lang1': 'English', 'lang2': 'Hindi', 'name': 'Name', 'other': 'Other'}
    words = sentence.split()
    combinedPredictions = {}
    for vectorizer, classifier in zip(vectorizers, classifiers):
        inputVectorized = vectorizer.transform(words)
        predictions = classifier.predict(inputVectorized)
        mappedPredictions = [labelMapping.get(label, 'Unknown') for label in predictions]
        for word, label in zip(words, mappedPredictions):
            if word not in combinedPredictions:
                combinedPredictions[word] = []
            combinedPredictions[word].append(label)
    finalPredictions = {word: max(set(labels), key=labels.count) for word, labels in combinedPredictions.items()}
    return finalPredictions

userSentence = input("Enter a sentence: ")
predictedLabels = predlanglabel(userSentence)
print("Prediction:")
for word, label in predictedLabels.items():
    print(f"{word}: {label}")