import sklearn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import nltk
import matplotlib as plt


class NewsArticleInstance:
    def __int__(self, keyword: list[str], text: str):
        self.label = keyword
        self.text = text


labels_counts = Counter()
labels = []
bigrams: list[defaultdict] = []
trigrams: list[defaultdict] = []
unigrams: list[defaultdict] = []
true_labels_bigrams: list[list[int]] = []
true_labels_trigrams: list[list[int]] = []
true_labels_unigrams: list[list[int]] = []
num_keywords = 5

# This section is meant to find the 5 most frequent labels
# Open and read file
df = pd.read_csv("data/news.csv", encoding="utf-8")
for keywords in df['Keywords']:
    try:
        labels_counts.update(keywords.split(','))
    except AttributeError:
        # there are some entries that are entry which leads to being read as a float instead of a string
        # these entries will be skipped.
        continue
# Using the most frequent labels, I can now filter the data to be more manageable
# filter the 5 most frequent labels
labels = sorted(labels_counts, key=labels_counts.get, reverse=True)[:num_keywords]


# Extracting features (unigram, bigrams, trigrams)
def update_unigrams(sentence: str, keyword_list: list[str]):
    try:
        result = defaultdict(int)
        tokens = nltk.word_tokenize(sentence)
        keywords = [0 for _ in range(len(labels))]
        for key in keyword_list:
            if key in labels:
                label_index = labels.index(key)
                keywords[label_index] = 1
            else:
                pass
        for index in range(len(tokens)):
            result[tokens[index]] += 1
        unigrams.append(result)
        true_labels_unigrams.append(keywords)
    except TypeError:
        pass


def update_bigrams(sentence: str, keyword_list: list[str]):
    try:
        result = defaultdict(int)
        tokens = nltk.word_tokenize(sentence)
        keywords = [0 for _ in range(len(labels))]
        for key in keyword_list:
            if key in labels:
                label_index = labels.index(key)
                keywords[label_index] = 1
            else:
                pass
        result[('<START>', tokens[0])] += 1
        for index in range(len(tokens) - 1):
            result[(tokens[index], tokens[index + 1])] += 1
        result[(tokens[-1], "<END>")] += 1
        bigrams.append(result)
        true_labels_bigrams.append(keywords)
    except TypeError:
        pass


def update_trigrams(sentence: str, keyword_list: list[str]):
    try:
        result = defaultdict(int)
        tokens = nltk.word_tokenize(sentence)
        keywords = [0 for _ in range(len(labels))]
        for key in keyword_list:
            try:
                label_index = labels.index(key)
                keywords[label_index] = 1
            except ValueError:
                pass
        result[('<START>', '<START>', tokens[0])] += 1
        result[('<START>', tokens[0], tokens[1])] += 1
        for index in range(len(tokens) - 1):
            result[(tokens[index], tokens[index + 1], tokens[index + 1])] += 1
        result[(tokens[-2], tokens[-1], "<END>")] += 1
        trigrams.append(result)
        true_labels_trigrams.append(keywords)
    except TypeError:
        pass


# running the functions
for i in range(len(df['Keywords'])):
    try:
        update_unigrams(df['Description'][i], df['Keywords'][i].split(","))
        update_bigrams(df['Description'][i], df['Keywords'][i].split(","))
        update_trigrams(df['Description'][i], df['Keywords'][i].split(","))
    except AttributeError:
        continue


# FUnctions Used
# Experiment Analysis Method
# instead calculating the accuracy using true negatives and true positives, this accuracy
# looks at whether at least one of the labels were predicted
def accuracy(true_vectors, predict_vectors, num_entries, pos):
    score = 0
    for i in range(num_entries):
        for j in range(num_keywords):
            if true_vectors[i][j] == pos and predict_vectors[i][j] == pos:
                score += 1
                break
    return round(score / num_entries*100, 2)


def data_split(X: list[defaultdict], y: list[list[int]]):
    dv = DictVectorizer(sparse=True)
    y = np.array(true_labels_unigrams)
    X = dv.fit_transform(unigrams)
    X, X_test, y, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.90, test_size=0.10)
    X_train, X_dev, y_train, y_dev = sklearn.model_selection.train_test_split(X, y, train_size=0.88, test_size=0.12)
    return X_train, y_train, X_dev, y_dev, X_test, y_test


# Splitting the data
# Unigrams
X_train_uni, y_train_uni, X_dev_uni, y_dev_uni, X_test_uni, y_test_uni = data_split(unigrams, true_labels_unigrams)
num_entries_uni = len(y_test_uni)

# Bigrams
X_train_bi, y_train_bi, X_dev_bi, y_dev_bi, X_test_bi, y_test_bi = data_split(bigrams, true_labels_bigrams)
num_entries_bi = len(y_test_bi)


# Trigrams
X_train_tri, y_train_tri, X_dev_tri, y_dev_tri, X_test_tri, y_test_tri = data_split(trigrams, true_labels_trigrams)
num_entries_tri = len(y_test_tri)
print("Training size:",len(y_train_tri))
print("Dev size:", len(y_dev_tri))
print("Test size:", len(y_test_tri))

# Logistic Regression

print("Starting Logistic Regression Tuning:")
logistic_uni = {}
logistic_bi = {}
logistic_tri = {}
# Initial Training
clf_uni_log = MultiOutputClassifier(LogisticRegression(max_iter=10000))
clf_uni_log.fit(X_train_uni, y_train_uni)
clf_bi_log = MultiOutputClassifier(LogisticRegression(max_iter=10000))
clf_bi_log.fit(X_train_bi, y_train_bi)
clf_tri_log = MultiOutputClassifier(LogisticRegression(max_iter=10000))
clf_tri_log.fit(X_train_tri, y_train_tri)
# Tuning
for solver in ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]:
    for C in [100, 10, 1.0, 0.1, 0.01]:
        params = {"solver": solver, "C": C}
        clf_uni_log.estimator.set_params(**params)
        prediction_log_uni = clf_uni_log.predict(X_dev_uni)

        clf_bi_log.estimator.set_params(**params)
        prediction_log_bi = clf_bi_log.predict(X_dev_bi)

        clf_tri_log.estimator.set_params(**params)
        prediction_log_tri = clf_tri_log.predict(X_dev_tri)

        logistic_uni[solver, C] = accuracy(prediction_log_uni, y_dev_uni, num_entries_uni, 1)
        print(solver, C, accuracy(prediction_log_uni, y_dev_uni, num_entries_uni, 1))
        logistic_bi[solver, C] = accuracy(prediction_log_bi, y_dev_bi, num_entries_bi, 1)
        print(solver, C, accuracy(prediction_log_bi, y_dev_bi, num_entries_bi, 1))
        logistic_tri[solver, C] = accuracy(prediction_log_tri, y_dev_tri, num_entries_tri, 1)
        print(solver, C, accuracy(prediction_log_tri, y_dev_tri, num_entries_tri, 1))

        # logistic_uni[solver, C] = accuracy_score(prediction_log_uni, y_dev_uni)
        # print(solver, C, accuracy_score(prediction_log_uni, y_dev_uni))
        # logistic_bi[solver, C] = accuracy_score(prediction_log_bi, y_dev_bi)
        # print(solver, C, accuracy_score(prediction_log_bi, y_dev_bi))
        # logistic_tri[solver, C] = accuracy_score(prediction_log_tri, y_dev_tri)
        # print(solver, C, accuracy_score(prediction_log_tri, y_dev_tri))
logistic_max_hyper_uni = max(logistic_uni)
logistic_max_hyper_bi = max(logistic_bi)
logistic_max_hyper_tri = max(logistic_tri)
print("These are the max hyperparameters:")
print(logistic_max_hyper_uni)
print(logistic_max_hyper_bi)
print(logistic_max_hyper_tri)

params = {"solver": logistic_max_hyper_tri[0], "C": logistic_max_hyper_tri[1]}

clf_uni_log.estimator.set_params(**params)
prediction_uni_log = clf_uni_log.predict(X_test_uni)

clf_bi_log.estimator.set_params(**params)
prediction_bi_log = clf_bi_log.predict(X_test_bi)

clf_tri_log.estimator.set_params(**params)
prediction_tri_log = clf_tri_log.predict(X_test_tri)


########################################################################################
# Random Forest
print("Starting Random Forest Tuning:")
forest_uni = {}
forest_bi = {}
forest_tri = {}
# Training
clf_uni_forest = MultiOutputClassifier(RandomForestClassifier())
clf_uni_forest.fit(X_train_uni, y_train_uni)
clf_bi_forest = MultiOutputClassifier(RandomForestClassifier())
clf_bi_forest.fit(X_train_bi, y_train_bi)
clf_tri_forest = MultiOutputClassifier(RandomForestClassifier())
clf_tri_forest.fit(X_train_tri, y_train_tri)
# Tuning
clf_tri_log.fit(X_train_tri, y_train_tri)
for feat in [10, 100, 1000]:
    for n in range(1, 20+1):
        print(feat, n)
        params = {"max_features": feat, "n_estimators": n}
        clf_uni_forest.estimator.set_params(**params)
        prediction_forest_uni = clf_uni_forest.predict(X_dev_uni)

        clf_bi_forest.estimator.set_params(**params)
        prediction_forest_bi = clf_bi_forest.predict(X_dev_bi)

        clf_tri_forest.estimator.set_params(**params)
        prediction_forest_tri = clf_tri_forest.predict(X_dev_tri)

        forest_uni[feat, n] = accuracy(prediction_forest_uni, y_dev_uni, num_entries_uni, 1)
        print(feat, n, accuracy(prediction_forest_uni, y_dev_uni, num_entries_uni, 1))
        forest_bi[feat, n] = accuracy(prediction_forest_bi, y_dev_bi, num_entries_bi, 1)
        print(feat, n, accuracy(prediction_forest_bi, y_dev_bi, num_entries_bi, 1))
        forest_tri[feat, n] = accuracy(prediction_forest_tri, y_dev_tri, num_entries_tri, 1)
        print(feat, n, accuracy(prediction_forest_tri, y_dev_tri, num_entries_tri, 1))
        
        # forest_uni[feat, n] = accuracy_score(prediction_forest_uni, y_dev_uni)
        # print(feat, n, accuracy_score(prediction_forest_uni, y_dev_uni))
        # forest_bi[feat, n] = accuracy_score(prediction_forest_bi, y_dev_bi)
        # print(feat, n, accuracy_score(prediction_forest_bi, y_dev_bi))
        # forest_tri[feat, n] = accuracy_score(prediction_forest_tri, y_dev_tri)
        # print(feat, n, accuracy_score(prediction_forest_tri, y_dev_tri))
forest_max_hyper_uni = max(forest_uni)
forest_max_hyper_bi = max(forest_bi)
forest_max_hyper_tri = max(forest_tri)
print("These are the max hyperparameters:")
print(forest_max_hyper_uni)
print(forest_max_hyper_bi)
print(forest_max_hyper_tri)

params = {"max_features": forest_max_hyper_uni[0], "n_estimators": forest_max_hyper_uni[1]}

clf_uni_forest.estimator.set_params(**params)
prediction_uni_forest = clf_uni_forest.predict(X_test_uni)

clf_bi_forest.estimator.set_params(**params)
prediction_bi_forest = clf_bi_forest.predict(X_test_bi)

clf_tri_forest.estimator.set_params(**params)
prediction_tri_forest = clf_tri_forest.predict(X_test_tri)

print("Logisitic Classification")
print("base accuracy")
print(round(accuracy_score(prediction_uni_log, y_test_uni)*100,2))
print(round(accuracy_score(prediction_bi_log, y_test_bi)*100,2))
print(round(accuracy_score(prediction_tri_log, y_test_tri)*100,2))

print("revised accuracy (pos=1):")
print(accuracy(prediction_uni_log, y_test_uni, num_entries_uni, 1))
print(accuracy(prediction_bi_log, y_test_bi, num_entries_bi, 1))
print(accuracy(prediction_tri_log, y_test_tri, num_entries_tri, 1))

print("revised accuracy (pos=0):")
print(accuracy(prediction_uni_log, y_test_uni, num_entries_uni, 0))
print(accuracy(prediction_bi_log, y_test_bi, num_entries_bi, 0))
print(accuracy(prediction_tri_log, y_test_tri, num_entries_tri, 0))

print("#####################################################################################\n")
print("Forest Classification")
print("base accuracy")
print(round(accuracy_score(prediction_uni_forest, y_test_uni)*100,2))
print(round(accuracy_score(prediction_bi_forest, y_test_bi)*100,2))
print(round(accuracy_score(prediction_tri_forest, y_test_tri)*100, 2))

print("revised accuracy (pos=1):")
print(accuracy(prediction_uni_forest, y_test_uni, num_entries_uni, 1))
print(accuracy(prediction_bi_forest, y_test_bi, num_entries_bi, 1))
print(accuracy(prediction_tri_forest, y_test_tri, num_entries_tri, 1))

print("revised accuracy (pos=0):")
print(accuracy(prediction_uni_forest, y_test_uni, num_entries_uni, 0))
print(accuracy(prediction_bi_forest, y_test_bi, num_entries_bi, 0))
print(accuracy(prediction_tri_forest, y_test_tri, num_entries_tri, 0))