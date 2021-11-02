import os
import numpy as np
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import urlextract
from collections import Counter
import re
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline

# set the paths for the folders
SPAM_PATH = os.path.join("data", "spam")
HAM_PATH = os.path.join("data", "ham")

# read the names of the files from each path
ham_filenames = [name for name in sorted(os.listdir(HAM_PATH)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_PATH)) if len(name) > 20]

import email
import email.policy

# write a function that parses the emails
def load_email(is_spam, filename, email_path="data"):
    directory = "spam" if is_spam else "ham"
    with open(os.path.join(email_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

# read the parsed emails into arrays
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
# look at one email
print(ham_emails[1].get_content().strip())

# create the test and train data sets
X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(spam_emails[1]["Subject"])


def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content().strip()
        except:
            content = str(part.get_payload()).strip()
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        soup = BeautifulSoup(html)
        return soup.getText().strip()

class EmailtoWordCounter(BaseEstimator, TransformerMixin):
    def __init__(self, lower_case = True):
        self.lower_case = lower_case

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            # replace urls by the string URL
            url_extractor = urlextract.URLExtract()
            urls = list(set(url_extractor.find_urls(text)))
            for url in urls:
                text = text.replace(url, "URL")
            # replace numbers
            text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            # remove punctuation
            text = re.sub(r'\W+', ' ', text, flags=re.M)
            # remove non-alphabet chars
            text = re.sub('[^a-zA-Z]', ' ', text)
            word_counts = Counter(text.split())
            X_transformed.append(word_counts)
        return np.array(X_transformed)

class WordCounterToVector(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
        self.vocabulary_ = {}

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
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
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))

X_few = X_train[:3]
X_few_wordcounts = EmailtoWordCounter().fit_transform(X_few)
vocab_transformer = WordCounterToVector(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
X_few_vectors.toarray()

preprocess_pipeline = Pipeline([
    ("email_to_word_count", EmailtoWordCounter()),
    ("wordcount_to_vector", WordCounterToVector())
])
X_train_transformed = preprocess_pipeline.fit_transform(X_train)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_clf = LogisticRegression(max_iter=1000, random_state=42)
score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3, verbose=3)
score.mean()