import numpy as np
import pandas as pd
from collections import defaultdict
from math import log
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Load dataset
data = pd.read_csv("D:/Harshitha docs/machine learning/data.csv")
# Split the dataset into training and testing sets and reset indices
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
# Implement the train_multinomial_nb function
def train_multinomial_nb(C, D):
    V = set()
    for sentence in D['Sentence']:
        V.update(sentence.split())

    N = len(D)
    prior = {}
    condprob = defaultdict(lambda: defaultdict(float))

    for c in C:
        Dc = D[D['Sentiment'] == c]
        Nc = len(Dc)
        prior[c] = Nc / N
        textc = ' '.join(Dc['Sentence'])
        
        Tct = defaultdict(int)
        for term in textc.split():
            Tct[term] += 1
        
        total_count = sum(Tct.values()) + len(V)  # Laplace smoothing
        
        for t in V:
            condprob[t][c] = (Tct[t] + 1) / total_count

    return V, prior, condprob
# Implement the apply_multinomial_nb function
def apply_multinomial_nb(C, V, prior, condprob, d):
    W = d.split()
    score = {}

    for c in C:
        score[c] = log(prior[c])
        for t in W:
            if t in V:
                score[c] += log(condprob[t][c])

    return max(score, key=score.get)
# Train the model
C = train_df['Sentiment'].unique()
V, prior, condprob = train_multinomial_nb(C, train_df)
test_sentences = test_df["Sentence"].tolist()
test_labels = test_df["Sentiment"].tolist()
# Evaluate the model
predictions = [apply_multinomial_nb(C, V, prior, condprob, sentence) for sentence in test_sentences]

# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy:.2f}")