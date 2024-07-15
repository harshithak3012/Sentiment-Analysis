import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math

# Load dataset
data = pd.read_csv(r"D:\Harshitha docs\machine learning\data.csv")

# Split the dataset into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Function to extract vocabulary
def extract_vocabulary(data):
    vocabulary = set()
    for sentence in data["Sentence"]:
        for word in sentence.split():
            vocabulary.add(word.lower())
    return vocabulary

# Function to count documents
def count_docs(data):
    return len(data)

# Function to count documents in each class
def count_docs_in_class(data, c):
    return len(data[data["Sentiment"] == c])

# Function to count documents in class containing term
def count_docs_in_class_containing_term(data, c, t):
    return sum(1 for sentence in data[data["Sentiment"] == c]["Sentence"] if t in sentence.split())

# Train Bernoulli Naive Bayes
def train_bernoulli_nb(data):
    V = extract_vocabulary(data)
    N = count_docs(data)
    C = data["Sentiment"].unique()
    prior = {}
    condprob = defaultdict(dict)
    
    for c in C:
        Nc = count_docs_in_class(data, c)
        prior[c] = Nc / N
        for t in V:
            Nct = count_docs_in_class_containing_term(data, c, t)
            condprob[t][c] = (Nct + 1) / (Nc + 2)
    
    return V, prior, condprob

# Apply Bernoulli Naive Bayes
def apply_bernoulli_nb(C, V, prior, condprob, d):
    Vd = set(d.split())
    score = {}
    
    for c in C:
        score[c] = math.log(prior[c])
        for t in V:
            if t in Vd:
                score[c] += math.log(condprob[t][c])
            else:
                score[c] += math.log(1 - condprob[t][c])
    
    return max(score, key=score.get)

# Training
V, prior, condprob = train_bernoulli_nb(train_data)

# Testing
test_sentences = test_data["Sentence"].tolist()
test_labels = test_data["Sentiment"].tolist()

predictions = [apply_bernoulli_nb(train_data["Sentiment"].unique(), V, prior, condprob, sentence) for sentence in test_sentences]

# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy:.2f}")
