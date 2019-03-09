
from bs4 import BeautifulSoup
import urllib.request
from nltk.stem import WordNetLemmatizer
import nltk
import sklearn
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
import nltk.chunk
#
#url = "https://en.wikipedia.org/wiki/Google"
#content = urllib.request.urlopen(url)
#soup = BeautifulSoup(content, "html.parser")
#f= open("input.txt","w+")
#for hit in soup.findAll(attrs={'class' : 'mw-body-content'}):
    #print(hit.text)
#    f.write(hit.text)

#nltk.download()
#f.close()

file = open("input.txt").read()

stokens = nltk.sent_tokenize(file)
wtokens = nltk.word_tokenize(file)


print("********shows sent and word tokenization*************")
n = 0
for s in stokens:
    n+=1
    print(s, "\n")
    if n > 6:
        break
n = 0
for t in wtokens:
    n+=1
    print(t)
    if n > 6:
        break


print("********shows POS of word tokens*************")

print(nltk.pos_tag(wtokens), "\n")

print("********shows Stem of word tokens*************")
n =0
for t in wtokens:
    n+=1
    pStemmer = PorterStemmer()
    print(pStemmer.stem(t))
    lStemmer = LancasterStemmer()
    print(lStemmer.stem(t))
    sStemmer = SnowballStemmer('english')
    print(sStemmer.stem(t))
    if n > 20:
        break


print("********shows Lemmatization of word tokens*************")

n =0
lemmatizer = WordNetLemmatizer()
for t in wtokens:
    n+=1

    print(t, ">>>>>" , lemmatizer.lemmatize(t))
    if n > 100:
        break


print("********shows Trigram of word tokens*************")






print("********shows NER of word tokens*************")
n = 0
for s in stokens:
    n+=1
    print(ne_chunk(pos_tag(wordpunct_tokenize(s))), "\n")
    if n > 6:
        break










