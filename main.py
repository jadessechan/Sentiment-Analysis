import re
import string
from collections import Counter
import nltk
import unicodedata
import pandas as pd
import plotly.express as px
import numpy as np


def main():
    """
        STEP 0:
            perform EDA (exploratory data analysis),
            plot and visualize main attributes of dataset

    """
    print("step 0...")

    df = pd.read_csv('corpora/imdb.csv')
    df.head()

    # show positive and negative review counts:
    fig = px.histogram(df, x="sentiment")
    fig.update_traces(marker_color="pink",marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5)
    fig.update_layout(title_text='IMDB Review Sentiment')
    fig.show()

    """
        STEP 1: 
        split data for training and testing,
        distinguish between positive and negative reviews

    """
    print("step 1...")

    # split the dataset for training and testing
    # 90% delegated for training and 10% for testing
    index = df.index
    df['random_number'] = np.random.randn(len(index))
    train = df[df['random_number'] <= 0.9]
    test = df[df['random_number'] > 0.1]

    neg_text = get_text(train, "negative")
    pos_text = get_text(train, "positive")

    """
        STEP 2: 
        pre-process the training data,
        make 2 BOW models for each sentiment

    """
    print("step 2...")

    # pre-process training data
    print("filtering...")
    neg_text = filter(neg_text)
    pos_text = filter(pos_text)
    print("cleaning...")
    neg_tokens = clean(neg_text)
    pos_tokens = clean(pos_text)

    # Generate word counts for each sentiment.
    neg_counts = bow_model(neg_tokens)
    pos_counts = bow_model(pos_tokens)

    """
        STEP 3: 
            compute the probabilities of each class occurring in the data
    """
    print("step 3...")

    neg_review_count = sentiment_count(train, "negative")
    pos_review_count = sentiment_count(train, "positive")

    # class probabilities (P(sentiment)).
    prob_positive = pos_review_count / len(train)
    prob_negative = neg_review_count / len(train)

    """
        STEP 4:
            predict on training set
            predict on the testing set
    """
    print("step 4...")

    # --- CAN COMMENT OUT WHEN DONE TRAINING ---

    # print("training...")
    # # get a sample review from training set
    # review = train.iat[1, 0]

    # # remove break tags (<br></br>) from review
    # review = re.sub('<.*?>', ' ', review)
    # print("TRAINING REVIEW: {0}".format(review))

    # # filter, clean, and tokenize the sample review
    # review = filter(review)
    # review = clean(review)

    # neg_pred = make_class_prediction(review, negative_counts, prob_negative, negative_review_count)
    # pos_pred = make_class_prediction(review, positive_counts, prob_positive, positive_review_count)
    # print("Negative prediction: {0}".format(neg_pred))
    # print("Positive prediction: {0}".format(pos_pred))

    # if neg_pred > pos_pred:
    #     print("Training prediction: NEGATIVE")
    # else:
    #     print("Training prediction: POSITIVE" + "\n")

    # ------------------------------------------

    print("testing...")

    # get a sample review from testing set
    test_review = test.iat[2, 0]

    # remove break tags (<br></br>) from review
    test_review = re.sub('<.*?>', ' ', test_review)
    print("TESTING REVIEW: {0}".format(test_review) + "\n")

    # filter, clean, and tokenize the sample review
    test_review = filter(test_review)
    tokenized_test_review = clean(test_review)

    # compute the negative and positive probabilities.
    neg_pred = make_class_prediction(tokenized_test_review, neg_counts, prob_negative, neg_review_count)
    pos_pred = make_class_prediction(tokenized_test_review, pos_counts, prob_positive, pos_review_count)

    # make decision based on which probability is greater.
    print("This review sentiment is...")
    if neg_pred > pos_pred:
        print("NEGATIVE")
    else:
        print("POSITIVE")


def get_text(reviews, score):
    """ Concatenate the reviews for a particular tone into 1 big string """
    r_list = ''
    for index, row in reviews.iterrows():
        if reviews.at[index, 'sentiment'] == score:
            r_list += row.loc['review']

    return r_list


def filter(text):
    """ Normalize text, remove unnecessary characters,
perform regex parsing, and make lowercase """

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


def clean(text):
    """ Remove stopwords, tokenize remaining words
    and perform lemmatization """

    stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    wnl = nltk.stem.WordNetLemmatizer()

    output = []
    for words in tokens:
        if words not in stopwords:
            # lemmatize words
            output.append(wnl.lemmatize(words))

    return output


def bow_model(text):
    """ make a bag of words for each sentiment """
    return Counter(text)


def sentiment_count(reviews, score):
    """ compute the count of each classification occurring in the data """

    class_count = 0
    for i, row in reviews.iterrows():
        if reviews.at[i, 'sentiment'] == score:
            class_count += 1

    return class_count


def make_class_prediction(tokens, counts, class_prob, class_count):
    """ compute the classification of each sentiment based on its probability in training set """

    prediction = 1
    text_counts = Counter(tokens)
    for word in text_counts:
        # get 'word' freq in the reviews for a given class, add 1 to smooth the value
        # add 1 smoothing prevents multiplying the prediction by 0 (in case 'word' is not in the training set)
        prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))

    return prediction * class_prob


main()
