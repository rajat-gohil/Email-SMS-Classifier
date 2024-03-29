# -*- coding: utf-8 -*-
"""I hope last classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qE_vRPdjiotw293Vkf-urS0dyiyypq7h
"""

# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from wordcloud import WordCloud
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Read the CSV file
encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
file_path = 'train.csv'

for encoding in encodings:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f'File Successfully read with encoding: {encoding}')
        break
    except UnicodeDecodeError:
        print(f"Failed to read with encoding: {encoding}")
        continue

if 'df' in locals():
    print("CSV file has been successfully loaded.")
else:
    print("All encoding attempts failed. Unable to read the CSV file.")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in set(string.punctuation)]
    tokens = [ps.stem(word) for word in tokens]
    transformed_text = " ".join(tokens)
    return transformed_text

df.rename(columns={'sms': 'text', 'label': 'target'}, inplace=True)
df['transformed_text'] = df['text'].apply(transform_text)

labels = ['1', '0']
plt.pie(df['target'].value_counts(), labels=labels, autopct="%0.2f")
plt.show()

df.head()

df.isnull().sum()

df.duplicated().sum()

df = df.drop_duplicates(keep='first')

df.duplicated().sum()

df['target'].value_counts()

import matplotlib.pyplot as plt

labels = ['1', '0']
plt.pie(df['target'].value_counts(), labels=labels, autopct="%0.2f")
plt.show()

import nltk

nltk.download('punkt')

df['num_characters'] =df['text'].apply(len)

df.head()

df['numwords'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

df.head()

df[['num_characters', 'numwords', 'num_sentences']].describe()

df[df['target'] == 0][['num_characters', 'numwords', 'num_sentences']].describe()

df.head()

import seaborn as sns
plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0 ] ['num_characters'])
sns.histplot(df[df['target'] == 1 ] ['num_characters'], color='red')

plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0 ] ['numwords'])
sns.histplot(df[df['target'] == 1 ] ['numwords'], color='red')

sns.pairplot(df,hue='target')

wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15, 6))
plt.imshow(spam_wc)

ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15, 6))
plt.imshow(ham_wc)

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

classifiers = {
    'SVC': SVC(kernel='sigmoid', gamma=1.0),
    'MultinomialNB': MultinomialNB(),
    'GaussianNB': GaussianNB(),
    'BernoulliNB': BernoulliNB(),
    'LogisticRegression': LogisticRegression(solver='liblinear', penalty='l1'),
    'DecisionTree': DecisionTreeClassifier(max_depth=5),
    'KNeighbors': KNeighborsClassifier(),
    'RandomForest': RandomForestClassifier(n_estimators=50, random_state=2),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=2),
    'Bagging': BaggingClassifier(n_estimators=50, random_state=2),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=50, random_state=2),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=2),
    'XGBoost': XGBClassifier(n_estimators=50, random_state=2)
}

def train_and_evaluate(classifiers, X_train, y_train, X_test, y_test):
    results = []
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        results.append({'Algorithm': name, 'Accuracy': accuracy, 'Precision': precision})
        print(f'{name}: Accuracy - {accuracy}, Precision - {precision}')
    return results

results = train_and_evaluate(classifiers, X_train, y_train, X_test, y_test)

results_df = pd.DataFrame(results)
results_df = pd.melt(results_df, id_vars='Algorithm')
sns.catplot(x='Algorithm', y='value', hue='variable', data=results_df, kind='bar', height=6)
plt.ylim(0.5, 1.0)
plt.xticks(rotation='vertical')
plt.show()

svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

estimators = [('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator = RandomForestClassifier()
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Stacking Classifier: Accuracy', accuracy_score(y_test, y_pred))
print('Stacking Classifier: Precision', precision_score(y_test, y_pred))

mnb.fit(X_train, y_train)

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))

