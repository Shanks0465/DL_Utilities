import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import twitter_samples
from nltk.probability import FreqDist

def initialize():
    nltk.download('punkt')
    nltk.download('stopwords')

def removelink(sentence):
    text=re.sub(r'https?:\/\/.*[\r\n]*', '', sentence)
    return text

def removehashtags(sentence):
    text=re.sub(r'#', '', sentence)
    return text

def tokenize(sentence):
    text=word_tokenize(sentence)
    return text

def tweettokenize(sentence):
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    text=tokenizer.tokenize(sentence)
    return text

def removestop(tokens):
    sequence=[]
    stopwords_english = stopwords.words('english')
    for word in tokens:
        if(word not in stopwords_english):
            sequence.append(word)
    return sequence

def removepunct(tokens):
    sequence=[]
    for word in tokens:
        if(word not in string.punctuation):
            sequence.append(word)
    return sequence

def stem(tokens):
    sequence=[]
    stemmer = PorterStemmer()
    for word in tokens:
        stem_word = stemmer.stem(word)
        sequence.append(stem_word)
    return sequence

def preprocess_tweet(tweet):
    tweet = removelink(tweet)
    tweet = removehashtags(tweet)
    tweet = tweettokenize(tweet)
    tweet = removestop(tweet)
    tweet = removepunct(tweet)
    tweet = stem(tweet)
    return tweet

def gettokens(sentences):
    tokens = []
    for sentence in sentences:
        sentence = preprocess_tweet(sentence)
        for word in sentence:
            tokens.append(word)
    return tokens


