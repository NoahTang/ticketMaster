#import modelTraining #import MODEL, VECTORIZER

import pickle

import pandas as pd
import numpy as np
import nltk
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
    #this is stupidly slow (BUT NOT REALLY)
    def __init__(self):
        self.keywords = ['billie eilish', 'olivia rodrigo']
        
        #self.techModel = pickle.load(open("models/tech.sav",'rb'))
        self.classModel = pickle.load(open("sniperTraining/class.sav", 'rb'))
        #self.techVectorizer = pickle.load(open("models/vectorizer_noTech.sav",'rb'))
        self.classVectorizer = pickle.load(open("sniperTraining/vectorizer.sav", 'rb'))

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        print("welcome to blackbox 1.5 ~ made for ticketMaster")
    
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

    def preProcess(self, phrase):
        
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

        phraseVectorized = self.classVectorizer.transform(processPhrase).toarray()
        phrase_df = pd.DataFrame(phraseVectorized)
        return phrase_df
     
    def kWordSearch(self, phrase): #not a dictionary anymore, 1D array.
        phrase = phrase.lower()
        for i in self.keywords: #the keywords
            if(i in phrase): #is found inside the phrase
                print(i)
                return True
        print("no_kClass") #"no keyword class"
        return False

    def snipe(self, phrase):
        if self.kWordSearch(phrase):
            print("KWORD_CATCH")
            return True
        prediction = self.classModel.predict(self.preProcess(phrase))
        if(prediction[0] == "scam"):
            print("MODEL_CATCH")
            return True
        else:
            return False


if __name__ == "__main__":
    testClass = sniper()
    pred = testClass.snipe("hello kids")
    print(pred)