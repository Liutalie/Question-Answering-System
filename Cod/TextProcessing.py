import json
from nltk.stem.snowball import SnowballStemmer
import spacy

# from deep_translator import GoogleTranslator

# def translateText(self):
#     with open('Dataset/Test.txt', 'r') as file:
#         self.m_testText = GoogleTranslator(source='en', target='ro').translate(file.read())
#     print(self.m_testText)


class TextProcessing:
    def __init__(self):
        self.m_stopWords = []

    def readStopWords(self):
        with open('stop_words_romanian.json', 'r', encoding='utf-8') as file:
            self.m_stopWords = json.load(file)

    def removeStopWords(self, query):
        words = query.split()
        newText = []
        for word in words:
            if word.lower() not in self.m_stopWords:
                newText.append(word.lower())
        return newText

    def stemmingOfWords(self, query):
        stemmer = SnowballStemmer('romanian')
        stemmed_words = [stemmer.stem(word) for word in query.split()]
        return stemmed_words

    def lemmaOfWords(self, query):
        lemma = spacy.load('ro_core_news_sm')
        doc = lemma(query)
        lemmatized_words = [token.lemma_ for token in doc]
        return lemmatized_words
