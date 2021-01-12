import sys
import pickle
import re
import nltk
import csv
import json
import twitter
from sklearn.feature_extraction.text import CountVectorizer  
import pandas as pd

from textblob import TextBlob
import re
import matplotlib.pyplot as plt
import string

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC


from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def pickling(filename: str, variable: object):
    with open(filename, "wb") as file:
        pickle.dump(variable, file)

stop_words = set(stopwords.words('english'))

def preprocess_tweet_text(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    
    return " ".join(filtered_words)

positif_total = 0
negatif_total = 0

currentSet = 0

training = []
test = []

charts_sum = []

config = json.load(open('config.json','r'))
	
api = twitter.Api(
    consumer_key=config['consumer_key'],
    consumer_secret=config['consumer_secret'],
    access_token_key=config['access_token'],
    access_token_secret=config['access_token_secret']
)

def percentage(part,whole):
    return 100 * float(part)/float(whole) 

csvFile = open("tweets_list.csv", 'w', encoding='utf-8')
csvWriter = csv.writer(csvFile)
csvWriter.writerow([
    "text", "created_at", "geo", "lang", "place", "coordinates",
    "user.favourites_count", "user.statuses_count", "user.description",
    "user.location", "user.id", "user.created_at", "user.verified",
    "user.following", "user.url", "user.listed_count",
    "user.followers_count", "user.default_profile_image",
    "user.utc_offset", "user.friends_count", "user.default_profile",
    "user.name", "user.lang", "user.screen_name", "user.geo_enabled",
    "user.profile_background_color", "user.profile_image_url",
    "user.time_zone", "id", "favorite_count", "retweeted", "source",
    "favorited", "retweet_count"
])

def createTestData(search_string):
    try:
        tweets_fetched=api.GetSearch(search_string, count=200)
        # This will return a list with twitter.Status objects. These have attributes for 
        # text, hashtags etc of the tweet that you are fetching. 
        # The full documentation again, you can see by typing pydoc twitter.Status at the 
        # command prompt of your terminal 
        print ("Great! We fetched "+str(len(tweets_fetched))+" tweets with the term "+search_string+"!!")

        
        # We will fetch only the text for each of the tweets, and since these don't have labels yet, 
        # we will keep the label empty 
        for status in tweets_fetched:

            #status.text = preprocess_tweet_text("" +status.text)
            
            csvWriter.writerow([
                    status.text, status.created_at, status.geo, status.lang,
                    status.place, status.coordinates,
                    status.user.favourites_count, status.user.statuses_count,
                    status.user.description, status.user.location,
                    status.user.id, status.user.created_at,
                    status.user.verified, status.user.following,
                    status.user.url, status.user.listed_count,
                    status.user.followers_count,
                    status.user.default_profile_image, status.user.utc_offset,
                    status.user.friends_count, status.user.default_profile,
                    status.user.name, status.user.lang,
                    status.user.screen_name, status.user.geo_enabled,
                    status.user.profile_background_color,
                    status.user.profile_image_url, status.user.time_zone,
                    status.id, status.favorite_count, status.retweeted,
                    status.source, status.favorited, status.retweet_count
                ])
    except:
        print ("Sorry there was an error!")
        return None
    

search_string=input("Bonjour, Que recherchons nous ? =>  ")
testData=createTestData(search_string)

def format_sentence(sent):
        return({word: True for word in nltk.word_tokenize(preprocess_tweet_text(sent))})

def train(set):

    pos = []
    with open("./data/" + set + "/pos_tweets.txt") as f:
        for i in f: 
            pos.append([format_sentence(i), 'pos'])

    neg = []
    with open("./data/" + set + "/neg_tweets.txt") as f:
        for i in f: 
            neg.append([format_sentence(i), 'neg'])


    training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
    test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]

    classifier = NaiveBayesClassifier.train(training)
    pickling("./pickles/"+set+"/original_naive_bayes.pickle", classifier)

    mnb_classifier = SklearnClassifier(MultinomialNB())
    mnb_classifier.train(training)
    pickling("./pickles/"+set+"/multinomial_naive_bayes.pickle", mnb_classifier)

    bnb_classifier = SklearnClassifier(BernoulliNB())
    bnb_classifier.train(training)
    pickling("./pickles/"+set+"/bernoulli_naive_bayes.pickle", bnb_classifier)

    lr_classifier = SklearnClassifier(LogisticRegression())
    lr_classifier.train(training)
    pickling("./pickles/"+set+"/logistic_regression.pickle", lr_classifier)

    SGD_classifier = SklearnClassifier(SGDClassifier())
    SGD_classifier.train(training)
    pickling("./pickles/"+set+"/sgd.pickle", SGD_classifier)

    linearSVC_classifier = SklearnClassifier(LinearSVC())
    linearSVC_classifier.train(training)
    pickling("./pickles/"+set+"/linear_svc.pickle", linearSVC_classifier)

    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training)
    pickling("./pickles/"+set+"/svc.pickle", SVC_classifier)

def launch(set):

    global charts_sum
    global currentSet

    if (set == "set1"):
        currentSet = 1
    else :
        currentSet = 2

    classifiers = {
        "NLTK Naive Bayes": "./pickles/"+set+"/original_naive_bayes.pickle",
        "Multinomial Naive Bayes": "./pickles/"+set+"/multinomial_naive_bayes.pickle",
        "Bernoulli Naive Bayes": "./pickles/"+set+"/bernoulli_naive_bayes.pickle",
        "Logistic Regression": "./pickles/"+set+"/logistic_regression.pickle",
        "LinearSVC": "./pickles/"+set+"/linear_svc.pickle",
        "SVC": "./pickles/"+set+"/svc.pickle",
        "SGDClassifier": "./pickles/"+set+"/sgd.pickle"
    }

    trained_classifiers = []
    for classifier in classifiers.values():
        with open(classifier, "rb") as fh:
            trained_classifiers.append(pickle.load(fh))

    for classifier in trained_classifiers:

        cs = pd.read_csv("tweets_list.csv")
        res = {}
        res["text"] = cs["text"].apply(format_sentence)

        positif = 0
        negatif = 0
        neutre = 0
        total_positif = []
        total_negatif = []
        total = 0

        for target in res["text"]:
            total += 1
            if(classifier.classify(target) == "pos"):
                positif += 1
            else :
                negatif += 1


        print(" <== START ==>")
        print(list(classifiers.keys())[trained_classifiers.index(classifier)])
        print("negatif a ", (negatif/total * 100), "%")
        print("positif a ", (positif/total * 100), "%")
        print("<== STOP ==> \n")

        global positif_total
        global negatif_total

        result = {
            "name" : list(classifiers.keys())[trained_classifiers.index(classifier)],
            "negatif" : (negatif/total * 100),
            "positif" : (positif/total * 100)
        }

        charts_sum.append(result)

        positif_total += (positif/total) / len(trained_classifiers)
        negatif_total += (negatif/total) / len(trained_classifiers)
    
    cs = pd.read_csv("tweets_list.csv")

    polarity_neg = 0
    polarity_neu = 0
    polarity_pos = 0
    polarity_total = 0

    for tweet in cs["text"]:
        score = SentimentIntensityAnalyzer().polarity_scores(tweet)
        polarity_neg += score['neg']
        polarity_neu += score['neu']
        polarity_pos += score['pos']

    print(polarity_neg)
    print(polarity_pos)
    print(polarity_neu)

    result = {
        "name" : "SentimentIntensityAnalyzer",
        "negatif" : (polarity_neg),
        "positif" : (polarity_pos),
        "neutre" : (polarity_neu)
    }

    
    charts_sum.append(result)



if "reload1" in sys.argv :
    print("learning from set1")
    train("set1")
    launch("set1")
elif "reload2" in sys.argv :
    print("learning from set2")
    train("set2")
    launch("set2")
else :
    if "launch1" in sys.argv :
        print("launching from set1 learning")
        launch("set1")
    else :
        print("launching from set2 learning")
        launch("set2")

def print_res() : 
    print(" \n \n <== TOTAL ==>")
    print("negatif a ", (negatif_total), "%")
    print("positif a ", (positif_total), "%")
    print("<== STOP ==> \n")

print_res()

print(charts_sum[0])
positive = charts_sum[0]['positif']
negative = charts_sum[0]['negatif']
colors = ['green','red']
labels = ['Positive ['+str(round(positive*100)/100)+'%]' , 'Negative ['+str(round(negative*100)/100)+'%]']
sizes = [positive, negative]
title = charts_sum[0]['name']
plt.subplot(3,3,1)
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title(title )
plt.axis('equal')

positive = charts_sum[1]['positif']
negative = charts_sum[1]['negatif']
colors = ['green','red']
labels = ['Positive ['+str(round(positive*100)/100)+'%]' , 'Negative ['+str(round(negative*100)/100)+'%]']
sizes = [positive, negative]
title = charts_sum[1]['name']
plt.subplot(3,3,2)
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title(title )
plt.axis('equal')

positive = charts_sum[2]['positif']
negative = charts_sum[2]['negatif']
colors = ['green','red']
labels = ['Positive ['+str(round(positive*100)/100)+'%]' , 'Negative ['+str(round(negative*100)/100)+'%]']
sizes = [positive, negative]
title = charts_sum[2]['name']
plt.subplot(3,3,3)
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title(title )
plt.axis('equal')

positive = charts_sum[3]['positif']
negative = charts_sum[3]['negatif']
colors = ['green','red']
labels = ['Positive ['+str(round(positive*100)/100)+'%]' , 'Negative ['+str(round(negative*100)/100)+'%]']
sizes = [positive, negative]
title = charts_sum[3]['name']
plt.subplot(3,3,4)
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title(title )
plt.axis('equal')

positive = charts_sum[4]['positif']
negative = charts_sum[4]['negatif']
colors = ['green','red']
labels = ['Positive ['+str(round(positive*100)/100)+'%]' , 'Negative ['+str(round(negative*100)/100)+'%]']
sizes = [positive, negative]
title = charts_sum[4]['name']
plt.subplot(3,3,5)
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title(title )
plt.axis('equal')

positive = charts_sum[5]['positif']
negative = charts_sum[5]['negatif']
colors = ['green','red']
labels = ['Positive ['+str(round(positive*100)/100)+'%]' , 'Negative ['+str(round(negative*100)/100)+'%]']
sizes = [positive, negative]
title = charts_sum[5]['name']
plt.subplot(3,3,6)
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title(title )
plt.axis('equal')

positive = charts_sum[6]['positif']
negative = charts_sum[6]['negatif']
colors = ['green','red']
labels = ['Positive ['+str(round(positive*100)/100)+'%]' , 'Negative ['+str(round(negative*100)/100)+'%]']
sizes = [positive, negative]
title = charts_sum[6]['name']
plt.subplot(3,3,8)
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title(title )
plt.axis('equal')

plt.suptitle(search_string +" tweeter sentiment with set : " + str(currentSet))
plt.show ()   # souvent optionnel, mais bon faisons bien les choses...



positive = positif_total
negative = negatif_total
colors = ['green','red']
labels = ['Positive ['+str(round(positive*10000)/100)+'%]' , 'Negative ['+str(round(negative*10000)/100)+'%]']
sizes = [positive, negative]
title = "Sum of the results"
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title(title )
plt.axis('equal')

plt.suptitle(search_string +" tweeter sentiment with set : " + str(currentSet))
plt.show ()  



positive = charts_sum[7]['positif']
negative = charts_sum[7]['negatif']
neutre = charts_sum[7]['neutre']
colors = ['green','grey','red']
labels = ['Positive ['+str(round(positive*100)/100)+'%]', 'Neutre ['+str(round(neutre*100)/100)+'%]' , 'Negatif ['+str(round(negative*100)/100)+'%]']
sizes = [positive,neutre, negative]
title = charts_sum[7]['name']
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title(title )
plt.axis('equal')

plt.suptitle(search_string +" tweeter sentiment with set : " + str(currentSet))
plt.show ()  