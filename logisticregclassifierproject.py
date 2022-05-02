import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import  nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

training_size =0.75
fstem = True #boolean for word stems
train= 0.8
rand_state = 5
stop_words = stopwords.words("english")  # gets stopwords
stemmer = SnowballStemmer("english")  # gets stems


def remove_stopwords(text):
    text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ',str(text).lower()).strip() #removes stopwords: Stop words are a set of commonly used words in a language that aren't useful in NLP
    if fstem:
        return " ".join([stemmer.stem(token) for token in text.split() if token not in stop_words])
    else:
        return " ".join([token for token in text.split() if token not in stop_words])


def main():
    tweets = pd.read_csv("twittersentiment.csv", encoding="ISO-8859-1", names = ["target", "ids", "date", "flag", "user", "text"], )
    tweets = tweets.sample(frac=0.1) # grabs first ~ 100,000 or so and shuffles them since they are in order by label
    labels = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"} #  labels
    tweets.target = tweets.target.apply(lambda x: labels[x]) #labels the data as pos, neg, neutral
    tweets.text = tweets.text.apply(remove_stopwords) # removes stop words
    vectorizer = TfidfVectorizer() # counts of words
    frequency = vectorizer.fit_transform(tweets.text) #frequency of words
    print("number of features: ", len(vectorizer.get_feature_names()))

    index = np.random.random(tweets.shape[0])
    X_train = frequency[index <= train, :]
    X_test = frequency[index> train, :]
    Y_train= tweets.target[index<= train]
    Y_test = tweets.target[index> train]
    tweets_train, tweets_test = train_test_split(tweets, test_size= 1-training_size, random_state= 30)
    print("train size : ", len(tweets_train))
    print("test size : ", len(tweets_test))

    clf = LogisticRegression(random_state=0, solver="saga", multi_class='multinomial').fit(X_train,Y_train) # logistic regression classifier
    Y_predict = clf.predict(X_test)  # predictions
    acc = sum(Y_predict == Y_test) / len(Y_test)  # accuray
    print("accuracy: ", acc)










# Calling main function
if __name__=="__main__":
    main()