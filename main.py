import re
import string
import csv
import random
from collections import Counter
import nltk
import unicodedata
import pandas as pd
import plotly.express as px
import numpy as np

def main():
    df = pd.read_csv('corpora/imdb.csv')
    df.head()

    # show positive and negative review counts:
    fig = px.histogram(df, x="sentiment")
    fig.update_traces(marker_color="pink",marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5)
    fig.update_layout(title_text='IMDB Review Sentiment')
    fig.show()

    # Pre-process and clean the data
    print("Filtering...")
    df['review'] = df['review'].apply(filter)
    print("Cleaning...")
    df['review'] = df['review'].apply(clean)
    df.head()

    # split the dataset for training and testing
    # 90% delegated for training and 10% for testing
    index = df.index
    df['random_number'] = np.random.randn(len(index))
    train = df[df['random_number'] <= 0.9]
    test = df[df['random_number'] > 0.1]

    negative_text = get_text(train['review'], 'negative')
    positive_text = get_text(train['review'], 'positive')

    negative_counts = count_text(negative_text)
    positive_counts = count_text(positive_text)

    print("Negative text sample: {0}".format(negative_text[:100]))
    print("Positive text sample: {0}".format(positive_text[:100]))

    # make bag of words model for positive and negative sentiments
def get_text(reviews, score):
    return " ".join(r[0].lower for r in reviews if r[1] == score)

def count_text(text):
    return Counter(text)


"""
    Normalize text, remove unnecessary characters, 
    perform regex parsing, and make lowercase
"""
def filter(text):
    # normalize text
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
    # replace html chars with ' '
    text = re.sub('<.*?>', ' ', text)
    # remove punctuation
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    # only alphabets and numerics
    text = re.sub('[^a-zA-Z]', ' ', text)
    # replace newline with space
    text = re.sub("\n", " ", text)
    # lower case
    text = text.lower()
    # split and join the words
    text = ' '.join(text.split())

    return text


"""
    Remove stopwords, tokenize remaining words
    and perform lemmatization
"""
def clean(text):
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    wnl = nltk.stem.WordNetLemmatizer()

    output = []
    for words in tokens:
        if words not in stopwords:
            # lemmatize words
            output.append(wnl.lemmatize(words))

    return output







# put reviews into one list
# d_list = []
# d_list.extend(train['review'].tolist())
# print(len(d_list))
# for i in range(3):
#     print(d_list[i])


# for list in d_list:
#     c = Counter(list)

# print(c.most_common(10))

def wordCount(text):
    # init a dictionary to store each word and its count
#     counts = {}
#     # loop through rows
#     for i in d_list:
#         for token in text:
#             if token not in counts.keys(): 
#                 counts[token] = 1
#             else: 
#                 counts[token] += 1
#     return counts

# word_freq = df['review'].apply(wordCount)

    counts = Counter()
    for token in text:
        counts[token] += 1
    return counts

# wordFreq = wordCount(d_list)
# print(wordFreq.most_common(10))
# list1 = ['x','y','z','x','x','x','y','z']
# print(Counter(list1))

main()