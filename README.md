# Sentiment-Analysis

This program uses Naive Bayes and Laplace smoothing to predict movie review sentiments based on the [imdb dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


*If the jupyter notebook (main.ipynb) is not rendering, visit [this link](https://nbviewer.jupyter.org/github/jadessechan/Sentiment-Analysis/blob/master/main.ipynb) OR visit [main.py](https://github.com/jadessechan/Sentiment-Analysis/blob/master/main.py) in this repo* 

## Getting started
1. clone or download this repository
```sh
git clone https://github.com/jadessechan/Sentiment-Analysis.git
```
2. open main.ipynb or main.py
3. run the code to see the final prediction for a movie review

## Demo
In real world scenarios, class imbalances are very common but for the purposes of this project I chose a balanced data set:
![image of imdb data histogram](https://github.com/jadessechan/Sentiment-Analysis/blob/master/imgs/imdb_histogram.png) <br />
The dataset is split equally between the positive and negative sentiment of 25,000 reviews each.
I used the plotly library to graph this data
```sh
import plotly.express as px
px.histogram()
```
Here is the final output:
![final prediction](https://github.com/jadessechan/Sentiment-Analysis/blob/master/imgs/final_output.png)

## Implementation
### Step 0: 
* perform EDA (exploratory data analysis)
* plot and visualize main attributes of dataset using the pandas dataframe
![dataframe of imdb data](https://github.com/jadessechan/Sentiment-Analysis/blob/master/imgs/imdb_dataframe.png)

### Step 1:
* split data for training and testing (90% delegated for training and 10% for testing)
 The sentiments are distributed randomly throughout the dataset, thankfully (less sorting to do)!
``` sh
index = df.index
df['random_number'] = np.random.randn(len(index))
train = df[df['random_number'] <= 0.9]
test = df[df['random_number'] > 0.1]
```
* distinguish between positive and negative reviews

### Step 2:
* pre-process the training data
* make 2 BOW models for each sentiment (Counter for python is a handy feature to quickly get a dictionary of *Keys* and its frequencies as *Values*)
```sh
from collections import Counter
Counter(text)
```
* The functions to filter and clean the text are the same as the ones in my [text prediciton program](https://github.com/jadessechan/Text-Prediction) used for regex parsing.
But unlike in the text prediction, I removed stopwords in order to ignore extraneous information
```sh
for words in tokens:
        if words not in stopwords:
            # lemmatize words
            output.append(wnl.lemmatize(words))
```

### Step 3:
* compute the probability of each class occurring in the data

### Step 4:
* predict on training set
* predict on the testing set <br />
This is where the magic happens, featuring the Naive Bayes algorithm and Laplace smoothing!🪄
```sh
def make_class_prediction(tokens, counts, class_prob, class_count):
    """ compute the classification of each sentiment based on its probability in training set """

    prediction = 1
    text_counts = Counter(tokens)
    for word in text_counts:
        # get 'word' freq in the reviews for a given class, add 1 to smooth the value
        # add 1 smoothing prevents multiplying the prediction by 0 (in case 'word' is not in the training set)
        prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))

    return prediction * class_prob
```
