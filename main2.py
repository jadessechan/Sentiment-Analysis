from collections import Counter
import unicodedata
import nltk
import csv
import pandas as pd
import numpy as np
import re
import string


def main():
    # read data
    # with open('corpora/imdb.csv', 'r') as file:
    #     reviews = list(csv.reader(file))

    df = pd.read_csv('corpora/imdb.csv')
    # print("Filtering...")
    # df['review'] = df['review'].apply(filter)
    # print("Cleaning...")
    # df['review'] = df['review'].apply(clean)

    # split the dataset for training and testing
    # 90% delegated for training and 10% for testing
    index = df.index
    df['random_number'] = np.random.randn(len(index))
    train = df[df['random_number'] <= 0.9]
    test = df[df['random_number'] > 0.1]

    positive_text = get_text(test, "positive")
    negative_text = get_text(test, "negative")

    # pre-process training data
    print("filtering...")
    positive_text = filter(positive_text)
    negative_text = filter(negative_text)

    # returns tokenized text
    print("cleaning...")
    pos_tokens = clean(positive_text)
    neg_tokens = clean(negative_text)

    # Generate word counts for negative tone.
    negative_counts = count_text(neg_tokens)
    # Generate word counts for positive tone.
    positive_counts = count_text(pos_tokens)

    # print("Negative text sample: {0}".format(negative_text[:100]))
    # print("Positive text sample: {0}".format(positive_text[:100]))
    # print(negative_counts.most_common(10))
    # print(positive_counts.most_common(10))

    positive_review_count = get_y_count(test, "positive")
    negative_review_count = get_y_count(test, "negative")
    print("NUM OF POS REVIEWS IN TEST: ", positive_review_count)
    print("NUM OF NEG REVIEWS IN TEST", negative_review_count)

    print("NUM OF REVIEWS IN TEST: ", len(test))

    # class probabilities (P(y)).
    # count_row = reviews.shape[0] - 1
    print("step 3...")
    prob_positive = positive_review_count / len(test)
    prob_negative = negative_review_count / len(test)

    # As you can see, we can now generate probabilities for which class a given review is part of.
    # The probabilities themselves aren't very useful -- we make our classification decision based on which value is
    # greater.
    print("predicting...")

    review = test.iat[1, 0]
    print("Review: {0}".format(review))
    # filter and clean the review
    review = filter(review)
    review = clean(review)

    neg_pred = make_class_prediction(review, negative_counts, prob_negative, negative_review_count)
    pos_pred = make_class_prediction(review, positive_counts, prob_positive, positive_review_count)
    print("Negative prediction: {0}".format(neg_pred))
    print("Positive prediction: {0}".format(pos_pred))

    if neg_pred > pos_pred:
        print("Sentiment prediction: NEGATIVE")
    else:
        print("Sentiment prediction: POSITIVE")


""" MEDIUM TUTORIAL 
from sklearn.feature_extraction.text import CountVectorizer
# bag of words method
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['review'])
test_matrix = vectorizer.transform(test['review'])

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# Split target and independent variables
X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']

# Fit model on data
lr.fit(X_train,y_train)

# Make predictions
predictions = lr.predict(X_test)

# find accuracy, precision, recall:
from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
confusion_matrix(predictions,y_test)

print(classification_report(predictions,y_test))
"""

def get_text(reviews, score):
    r_list = ''
    # r_list = []
    # Join together the text in the reviews for a particular tone.
    for index, row in reviews.iterrows():
        # row = Series, reviews = Dataframe
        if reviews.at[index, 'sentiment'] == score:
            # print(reviews.at[index, 'sentiment'] + " REV SENT: " + score)
            # convert to string
            # x = reviews.to_string(columns=['review'], header=False, index=False, index_names=False).split('\n')
            # r_list = [','.join(ele.split()) for ele in x]
            r_list += row.loc['review']
            # r_list.join(reviews['review'].iloc[index])
    # return: string
    return r_list


# param: list of strings (tokens)
def count_text(text):
    # words = re.split("\s+", text)
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
    Remove stopwords and profanity, tokenize remaining words,
    perform lemmatization and POS tagging (optional)
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
            # output.append(words)

    return output


"""
    STEP 3: 
    make predictions from the training data
    transform BOW freqs to probabilities
    compute the probabilities of each class occurring in the data

"""


# Compute the count of each classification occurring in the data.
# loop through reviews (test set) and check 'sentiment' col to count num of + and - reviews
def get_y_count(reviews, score):
    class_count = 0
    # # subtract 1 for the header
    # count_row = reviews.shape[0] - 1
    for index, row in reviews.iterrows():
        if reviews.at[index, 'sentiment'] == score:
            class_count += 1

    return class_count


#   return len([r for r in reviews if r[1] == str(score)])


def make_class_prediction(tokens, counts, class_prob, class_count):
    prediction = 1
    text_counts = Counter(tokens)
    for word in text_counts:
        # For every word in the text, we get the number of times that word occured in the reviews for a given class,
        # add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also
        # smooth the denominator). Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist
        # in the training data.We also smooth the denominator counts to keep things even.
        prediction *= text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))

        # print("Word: {0}, Prediction: {1}".format(word, prediction))

    # Now we multiply by the probability of the class existing in the documents.
    return prediction * class_prob


main()
