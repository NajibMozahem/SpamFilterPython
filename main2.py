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
import urlextract
from scipy.sparse import csr_matrix


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
import email
import email.policy
def load_email(is_spam, filename):
    directory = spam if is_spam else ham
    with open(os.path.join(directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

# now that we have the emails, split the data into test and train sets
X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails + [1] * len(spam_emails)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# now we need a function to convert the email into a string text
def email_to_text(email):
    content = ""
    main_type = email.get_content_maintype()
    if main_type == "multipart":
        for part in email.get_payload():
            if part.get_content_maintype() == "text":
                content = part.get_payload()
    elif main_type == "text":
        type = email.get_content_type()
        if type == "text/html":
            content = email.get_content()
            soup = BeautifulSoup(content)
            content = soup.getText()
        elif type == "text/plain":
            content = email.get_content()
    return content.strip()

# now I need a transformer that will generate word counts in each email
class EmailToWordCount(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email)
            # convert to lower case
            text = text.lower()
            # replace urls by URL
            url_extractor = urlextract.URLExtract()
            urls = list(set(url_extractor.find_urls(text)))
            for url in urls:
                text = text.replace(url, "URL")
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
X_few_matrix = word_counter_to_matrix.fit(X_few_word_counts)
X_few_matrix.toarray()
# seems fine

