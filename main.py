import os
import re
import urllib
from tarfile import open as open_tar
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import email
import email.policy


download_url = "https://spamassassin.apache.org/old/publiccorpus/"
download_dir = os.path.join("datasets", "downloads")
ham = os.path.join("datasets", "ham")
spam = os.path.join("datasets", "spam")
os.makedirs(download_dir, exist_ok=True)
os.makedirs(ham, exist_ok=True)
os.makedirs(spam, exist_ok=True)
file_names = {
    "20021010_easy_ham.tar.bz2": "ham",
    "20021010_hard_ham.tar.bz2": "ham",
    "20021010_spam.tar.bz2": "spam",
    "20030228_easy_ham.tar.bz2": "ham",
    "20030228_easy_ham_2.tar.bz2": "ham",
    "20030228_hard_ham.tar.bz2": "ham",
    "20030228_spam.tar.bz2": "spam",
    "20030228_spam_2.tar.bz2": "spam",
    "20050311_spam_2.tar.bz2": "spam"
}
#download the files and unzip them
for key, value in file_names.items():
    file_name = os.path.join(download_dir, key)
    urllib.request.urlretrieve(download_url + key, file_name)
    with open_tar(file_name) as tar:
        tar.extractall(path=download_dir)
# delete the zipped files
for key, value in file_names.items():
    os.remove(os.path.join(download_dir, key))
#move the unzipped emails into appropriate folders
for folder in os.listdir(download_dir):
    if "spam" in folder:
        target = "spam"
    elif "ham" in folder:
        target = "ham"
    for file in os.listdir(os.path.join(download_dir, folder)):
        old_name = os.path.join(download_dir, folder, file)
        new_name = os.path.join("datasets", target, file)
        os.replace(old_name, new_name)
# delete the downloads folder and all subfolders
shutil.rmtree(download_dir)

# now I read the emails into separate lists
ham_filenames = [name for name in os.listdir(ham) if len(name) > 20]
spam_filenames = [name for name in os.listdir(spam) if len(name) > 20]

# now read the emails into arrays
def load_email(is_spam, filename):
    directory = spam if is_spam else ham
    with open(os.path.join(directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

# now delete the emails
shutil.rmtree("datasets")

# now that we have the emails, split the data into test and train sets
X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# now we need a function to convert the email into a string text
def email_to_text(email):
    content = ""
    for part in email.walk():
        content_type = part.get_content_type()
        if content_type in ("text/plain", "text/html"):
            try:
                content = part.get_content()
            except:
                content = str(part.get_payload())
            if content_type == "text/html":
                soup = BeautifulSoup(content)
                content = soup.getText(content)
            return content.strip()
    return content

# now I need a transformer that will generate word counts in each email
class EmailToWordCount(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            # convert to lower case
            text = text.lower()
            # replace urls by URL
            text = re.sub(r'(https?://\S+)', "URL", text)
            #url_extractor = urlextract.URLExtract()
            #urls = list(set(url_extractor.find_urls(text)))
            #for url in urls:
                #text = text.replace(url, "URL")
            # replace numbers by NUMBER
            text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            # replace emails by EMAIL
            text = re.sub('\S*@\S*\s?', 'EMAIL', text)
            # remove punctuation
            text = re.sub(r'[\W+]', ' ', text)
            # remove non alphabet characters
            text = re.sub('[^a-zA-Z]', ' ', text)
            word_count = Counter(text.split())
            X_transformed.append(word_count)
        return np.array(X_transformed)

# try it out
X_few = X[0:3]
X_few_word_counts = EmailToWordCount().fit_transform(X_few)
X_few_word_counts
# seems fine

# now we need a transformer to create a matrix that represents all these counts
class WordCountToMatrix(BaseEstimator, TransformerMixin):

    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += count
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size+1))

# let us try it out
word_counter_to_matrix = WordCountToMatrix(vocabulary_size=10)
X_few_matrix = word_counter_to_matrix.fit_transform(X_few_word_counts)
X_few_matrix.toarray()
# seems fine

# run ML algorithms
# first transform all data

preprocess_pipeline = Pipeline([
    ("email_to_word_count", EmailToWordCount()),
    ("word_count_to_matrix", WordCountToMatrix())
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)

# Logistic regression
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(max_iter=10000)
score = cross_val_score(lr_clf, X_train_transformed, y_train, cv=3)
lr_accuracy = score.mean()
score = cross_val_score(lr_clf, X_train_transformed, y_train, cv=3, scoring="f1")
lr_f1 = score.mean()

# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

knn_clf = KNeighborsClassifier()
n_neighbors = range(2, 10)
parameters = dict(n_neighbors=n_neighbors)
knn_grid = GridSearchCV(knn_clf, parameters, cv=3)
best_model = knn_grid.fit(X_train_transformed, y_train)
best_parameters = best_model.best_params_
knn_clf = KNeighborsClassifier(n_neighbors=best_parameters["n_neighbors"])
knn_clf.fit(X_train_transformed, y_train)
knn_yhat = knn_clf.predict(X_train_transformed)
knn_accuracy = accuracy_score(y_train, knn_yhat)
knn_f1 = f1_score(y_train, knn_yhat)

# Decision tree
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
max_depth = list(range(1, 50, 5))
min_samples_split = list(range(1, 20, 2))
max_features = np.linspace(0.1, 0.9, 5)
parameters = dict(max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)
tree_grid = GridSearchCV(tree_clf, parameters, cv=3)
best_model = tree_grid.fit(X_train_transformed, y_train)
best_parameters = best_model.best_params_
tree_clf = DecisionTreeClassifier(max_depth=best_parameters["max_depth"],
                                  min_samples_split=best_parameters["min_samples_split"],
                                  max_features=best_parameters["max_features"])
tree_clf.fit(X_train_transformed, y_train)
tree_yhat = tree_clf.predict(X_train_transformed)
tree_accuracy = accuracy_score(y_train, tree_yhat)
tree_f1 = f1_score(y_train, tree_yhat)

models = ["logistic", "knn", "tree"]
accuracy = [lr_accuracy, knn_accuracy, tree_accuracy]
f1 = [lr_f1, knn_f1, tree_f1]

plt.plot(models, accuracy, label="accuracy")
plt.plot(models, f1, label="f1 score")
plt.legend()