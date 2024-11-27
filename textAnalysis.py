import pickle

import pandas as pd
import numpy as np
import nltk
import json
import string
from array import array
import re
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#nltk.download('all', force=True)


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

# remove keyfault stopwords

class sniper:
    #this is stupidly slow
    def __init__(self):
            self.keywords = {'backend': ["backend", "back-end", "back end"],
                             'cloud': ["cloud", "azure"],
                             'data': ["data analyst", "data engineer", "data scientist"],
                             'databases': ["database", "data architect", "aws", "db"],
                             'hardware': ["embedded", "mechanical", "electrical", 
                                         "hardware", "photonics", "radio", "circuits"], 
                             'fullstack': ["full stack", "fullstack", "full-stack"],
                             'it_devops': ["devops", "i.t.", "it", "information technology"],
                             'mobile': ["android", "ios", "mobile"],
                             'networks': ["network", "data communication", "networks"],
                             'pm': ["project manager", "project coordinator", "pm"],
                             'qa': ["qa", "quality", "tester", "test"],
                             'security': ["security", "cyber"],
                             'systems': ["systems engineer", "system engineer"],
                             'ui_ux': ["frontend", "ui", "ux", "front-end"]}
            
            self.techModel = pickle.load(open("models/tech.sav",'rb'))
            self.classModel = pickle.load(open("models/class.sav", 'rb'))
            self.techVectorizer = pickle.load(open("models/vectorizer_noTech.sav",'rb'))
            self.classVectorizer = pickle.load(open("models/vectorizer.sav", 'rb'))

            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()

            print("welcome to blackbox 1.0")
    
    # Removes special characters
    def remove_special_characters(self, text):
        pattern = r'[^a-zA-Z\s]'
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text

    def remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def remove_stopwords(self,text):
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        return " ".join(filtered_text)

    def stem_words(self,text):
        word_tokens = word_tokenize(text)
        stems = [self.stemmer.stem(word) for word in word_tokens]
        return " ".join(stems)
    
    def lemma_words(self,text):
        word_tokens = word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in text]
        #print(lemmatized_words)
        return " ".join(lemmatized_words)
    
    def remove_whitespace(self,text):
        return " ".join(text.split())

    def preProcess(self, phrase, typ):
        
        processPhrase = phrase #change this later on
        
        #processPhrase = processPhrase.lower()
        processPhrase = self.remove_special_characters(processPhrase)
        processPhrase = self.remove_punctuation(processPhrase)
        processPhrase = self.remove_stopwords(processPhrase) 
        processPhrase = self.remove_whitespace(processPhrase) 
        
        processPhrase = self.stem_words(processPhrase)
        #print(processPhrase)
        #processPhrase = self.lemma_words(processPhrase) 

        #print(type(processPhrase))

        #processPhrase = ' '.join(processPhrase)
        processPhrase = [processPhrase]

        #print(processPhrase)

        if typ == "class":
            phraseVectorized = self.classVectorizer.transform(processPhrase).toarray()
            phrase_df = pd.DataFrame(phraseVectorized)
        elif typ == "tech":
            phraseVectorized = self.techVectorizer.transform(processPhrase).toarray()
            phrase_df = pd.DataFrame(phraseVectorized)

        return phrase_df
     
    '''def kWordSearch(self, title):
        title = title.lower()
        for i in self.keywords: #the keyword categories
            for j in range(len(self.keywords[i])): #goes through the array belonging to the categories
                if((self.keywords[i][j]) in title): #is found inside the phrase
                    return i
                
        return "no_kClass" #"no keyword class"'''

    def snipe(self, phrase):
        prediction = self.classModel.predict(self.preProcess(phrase,"class"))
        return prediction[0]