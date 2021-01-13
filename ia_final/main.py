import sys
import pickle
import re
import nltk
from tkinter import *
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

# Méthode pour enregirster une machine learning pré entrainée
def pickling(filename: str, variable: object): 
    with open(filename, "wb") as file:
        pickle.dump(variable, file)

# Utilisation de l'anglais uniquement
stop_words = set(stopwords.words('english'))

# Nettoyer les tweets pour de meilleurs performances
def preprocess_tweet_text(tweet):
    # n'utiliser que des lowercases
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    # Some words do not contribute much to the machine learning model, 
    # so it's good to remove them. A list of stopwords can be defined by the nltk library, or it can be business-specific.
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
    # Eliminating affixes (circumfixes, suffixes, prefixes, infixes) 
    # from a word in order to obtain a word stem.
    # Porter Stemmer is the most widely used technique because it is very fast. 
    # Generally, stemming chops off end of the word, and mostly it works fine.
    # Example: Working -> Work
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]

    # The goal is same as with stemming, but stemming 
    # a word sometimes loses the actual meaning of the word. 
    # Lemmatization usually refers to doing things properly using 
    # vocabulary and morphological analysis of words. 
    # It returns the base or dictionary form of a word, also known as the lemma .
    # Example: Better -> Good.
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    
    return " ".join(filtered_words)

positif_total = 0
negatif_total = 0

currentSet = 0

training = []
test = []

charts_sum = []

search_string=""

def percentage(part,whole):
    return 100 * float(part)/float(whole) 

config = json.load(open('config.json','r'))

# use API creds - currently using creds that I found online
api = twitter.Api(
    consumer_key=config['consumer_key'],
    consumer_secret=config['consumer_secret'],
    access_token_key=config['access_token'],
    access_token_secret=config['access_token_secret']
)

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

def format_sentence(sent):
        # A tokenizer that divides a string into substrings by splitting on the specified string (defined in subclasses).
        return({word: True for word in nltk.word_tokenize(preprocess_tweet_text(sent))})

def train(set):

    # in function of the selected data set we train the model
    pos = []
    with open("./data/" + set + "/pos_tweets.txt") as f:
        for i in f: 
            pos.append([format_sentence(i), 'pos'])

    neg = []
    with open("./data/" + set + "/neg_tweets.txt") as f:
        for i in f: 
            neg.append([format_sentence(i), 'neg'])

    # use the other set to verify the model
    oppositeSet = ""
    if (set == "set1"):
        oppositeSet = "set2"
    else :
        oppositeSet = "set1"

    pos_verify = []
    with open("./data/" + oppositeSet + "/pos_tweets.txt") as f:
        for i in f: 
            pos_verify.append([format_sentence(i), 'pos'])

    neg_verify = []
    with open("./data/" + oppositeSet + "/neg_tweets.txt") as f:
        for i in f: 
            neg_verify.append([format_sentence(i), 'neg'])


    training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
    test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]
    verify = pos_verify[0:] + neg_verify[0:]

    learning = [["Name","Test","Verify"]]
    classifier = NaiveBayesClassifier.train(training)
    pickling("./pickles/"+set+"/original_naive_bayes.pickle", classifier)
    print("\n** NaiveBayesClassifier Training")
    print("Accuracy percent test : \n", str((nltk.classify.accuracy(classifier, test)) * 100))
    print("Accuracy percent verify : \n", str((nltk.classify.accuracy(classifier, verify)) * 100))
    print("**********************************")
    learning.append(["NaiveBayesClassifier Training",str(round(nltk.classify.accuracy(classifier, test)*10000) / 100),str(round(nltk.classify.accuracy(classifier, verify)*10000) / 100)])

    mnb_classifier = SklearnClassifier(MultinomialNB())
    mnb_classifier.train(training)
    pickling("./pickles/"+set+"/multinomial_naive_bayes.pickle", mnb_classifier)
    print("\n** MultinomialNB Classifier Training")
    print("Accuracy percent test : \n", str((nltk.classify.accuracy(mnb_classifier, test)) * 100))
    print("Accuracy percent verify : \n", str((nltk.classify.accuracy(mnb_classifier, verify)) * 100))
    print("**********************************")
    learning.append(["MultinomialNB Classifier Training",str(round(nltk.classify.accuracy(mnb_classifier, test)*10000) / 100),str(round(nltk.classify.accuracy(mnb_classifier, verify)*10000) / 100)])

    bnb_classifier = SklearnClassifier(BernoulliNB())
    bnb_classifier.train(training)
    pickling("./pickles/"+set+"/bernoulli_naive_bayes.pickle", bnb_classifier)
    print("\n** BernoulliNB Classifier Training")
    print("Accuracy percent test : \n", str((nltk.classify.accuracy(bnb_classifier, test)) * 100))
    print("Accuracy percent verify : \n", str((nltk.classify.accuracy(bnb_classifier, verify)) * 100))
    print("**********************************")
    learning.append(["BernoulliNB Classifier Training",str(round(nltk.classify.accuracy(bnb_classifier, test)*10000) / 100),str(round(nltk.classify.accuracy(bnb_classifier, verify)*10000) / 100)])


    lr_classifier = SklearnClassifier(LogisticRegression())
    lr_classifier.train(training)
    pickling("./pickles/"+set+"/logistic_regression.pickle", lr_classifier)
    print("\n** LogisticRegression Classifier Training")
    print("Accuracy percent test : \n", str((nltk.classify.accuracy(lr_classifier, test)) * 100))
    print("Accuracy percent verify : \n", str((nltk.classify.accuracy(lr_classifier, verify)) * 100))
    print("**********************************")
    learning.append(["LogisticRegression Classifier Training",str(round(nltk.classify.accuracy(lr_classifier, test)*10000) / 100),str(round(nltk.classify.accuracy(lr_classifier, verify)*10000) / 100)])


    SGD_classifier = SklearnClassifier(SGDClassifier())
    SGD_classifier.train(training)
    pickling("./pickles/"+set+"/sgd.pickle", SGD_classifier)
    print("\n** SGD_classifier Classifier Training")
    print("Accuracy percent test : \n", str((nltk.classify.accuracy(SGD_classifier, test)) * 100))
    print("Accuracy percent verify : \n", str((nltk.classify.accuracy(SGD_classifier, verify)) * 100))
    print("**********************************")
    learning.append(["SGD_classifier Classifier Training",str(round(nltk.classify.accuracy(SGD_classifier, test)*10000) / 100),str(round(nltk.classify.accuracy(SGD_classifier, verify)*10000) / 100)])


    linearSVC_classifier = SklearnClassifier(LinearSVC())
    linearSVC_classifier.train(training)
    pickling("./pickles/"+set+"/linear_svc.pickle", linearSVC_classifier)
    print("\n** linearSVC Classifier Training")
    print("Accuracy percent test : \n", str((nltk.classify.accuracy(linearSVC_classifier, test)) * 100))
    print("Accuracy percent verify : \n", str((nltk.classify.accuracy(linearSVC_classifier, verify)) * 100))
    print("**********************************")
    learning.append(["linearSVC Classifier Training",str(round(nltk.classify.accuracy(linearSVC_classifier, test)*10000) / 100),str(round(nltk.classify.accuracy(linearSVC_classifier, verify)*10000) / 100)])


    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training)
    pickling("./pickles/"+set+"/svc.pickle", SVC_classifier)
    print("\n** SVC Classifier Training")
    print("Accuracy percent test : \n", str((nltk.classify.accuracy(SVC_classifier, test)) * 100))
    print("Accuracy percent verify : \n", str((nltk.classify.accuracy(SVC_classifier, verify)) * 100))
    print("**********************************")
    learning.append(["SVC Classifier Training",str(round(nltk.classify.accuracy(SVC_classifier, test)*10000) / 100),str(round(nltk.classify.accuracy(SVC_classifier, verify)*10000) / 100)])

    # display the training table
    root = Tk(className='Training Stats') 
    # set window size
    for i in range(8): 
            for j in range(3): 
                root.e = Entry(relief=GROOVE,width=30)
                root.e.grid(row=i, column=j) 
                root.e.insert(END, learning[i][j]) 
    root.mainloop() 



def launch(set):

    global search_string
    # ask for research
    search_string=input("Bonjour, Que recherchons nous ? =>  ")
    testData=createTestData(search_string)

    global charts_sum
    global currentSet

    if (set == "set1"):
        currentSet = 1
    else :
        currentSet = 2

    # get our trained classifiers
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

        # use the tweets already find to get the POS/NEG percentage
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
    
    # do the same as before but with : SentimentIntensityAnalyzer().polarity_scores(.....)
    cs = pd.read_csv("tweets_list.csv")

    polarity_neg = 0
    polarity_neu = 0
    polarity_pos = 0
    polarity_total = 0
    negatif_total_polarity = 0
    positif_total_polarity = 0

    for tweet in cs["text"]:
        score = SentimentIntensityAnalyzer().polarity_scores(tweet)
        # if only pos and neg
        if (score['neg'] >  score['pos']):
            negatif_total_polarity += 1
        elif (score['neg'] <  score['pos']): 
            positif_total_polarity += 1
        else : 
            negatif_total_polarity += 0.5
            positif_total_polarity += 0.5
        polarity_total += 1
        polarity_neg += score['neg']
        polarity_neu += score['neu']
        polarity_pos += score['pos']

    charts_sum.append({
        "name" : "SentimentIntensityAnalyzer POS/NEG",
        "negatif" : (negatif_total_polarity/polarity_total)* 100,
        "positif" : (positif_total_polarity/polarity_total)* 100
    })

    charts_sum.append({
        "name" : "SentimentIntensityAnalyzer",
        "negatif" : (polarity_neg),
        "positif" : (polarity_pos),
        "neutre" : (polarity_neu)
    })

if "reload1" in sys.argv :# train ? 
    print("\n --> learning from set1 <--")
    train("set1")
    launch("set1")
elif "reload2" in sys.argv :# train ? 
    print("\n --> learning from set2 <--")
    train("set2")
    launch("set2")
else : # use the app
    if "launch1" in sys.argv :
        print("\n --> launching from set1 learning <--")
        launch("set1")
    else :
        print("\n --> launching from set2 learning <--")
        launch("set2")

def print_res() : 
    print(" \n \n <== TOTAL ==>")
    print("negatif a ", (negatif_total), "%")
    print("positif a ", (positif_total), "%")
    print("<== STOP ==> \n")

print_res()


# ploting all
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

positive = charts_sum[8]['positif']
negative = charts_sum[8]['negatif']
neutre = charts_sum[8]['neutre']
colors = ['blue','grey','yellow']
labels = ['Positive ['+str(round(positive*100)/100)+'%]', 'Neutre ['+str(round(neutre*100)/100)+'%]' , 'Negatif ['+str(round(negative*100)/100)+'%]']
sizes = [positive,neutre, negative]
title = charts_sum[8]['name']
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title(title )
plt.axis('equal')

plt.suptitle(search_string +" tweeter sentiment with set : " + str(currentSet))
plt.show ()  


positive = charts_sum[7]['positif']
negative = charts_sum[7]['negatif']
colors = ['blue','yellow']
labels = ['Positive ['+str(round(positive*100)/100)+'%]', 'Negatif ['+str(round(negative*100)/100)+'%]']
sizes = [positive, negative]
title = charts_sum[7]['name']
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title(title )
plt.axis('equal')

plt.suptitle(search_string +" tweeter sentiment with set : " + str(currentSet))
plt.show ()  






plt.subplot(1,2,1)
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


plt.subplot(1,2,2)
positive = charts_sum[7]['positif']
negative = charts_sum[7]['negatif']
colors = ['blue','yellow']
labels = ['Positive ['+str(round(positive*100)/100)+'%]', 'Negatif ['+str(round(negative*100)/100)+'%]']
sizes = [positive, negative]
title = charts_sum[7]['name']
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title(title )
plt.axis('equal')

plt.suptitle(search_string +" tweeter sentiment final result with set : " + str(currentSet))
plt.show ()  