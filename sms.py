import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle

#reading the data

sms = pd.read_excel("spam.xlsx")

sms['label'] = sms['class'].map({'ham':0, 'spam':1})

X = sms['message']
y = sms['label']

# extract feature with CountVectorizer

cv = CountVectorizer()
X = X.dropna()
len(X)
X = cv.fit_transform(X)
