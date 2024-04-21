import json
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy


# from deep_translator import GoogleTranslator

# def translateText(self):
#     with open('Dataset/Test.txt', 'r') as file:
#         self.m_testText = GoogleTranslator(source='en', target='ro').translate(file.read())
#     print(self.m_testText)


class TextProcessing:
    def __init__(self):
        self.m_stopWords = []
        self.nlp = spacy.load('ro_core_news_sm')

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
        lemma = self.nlp
        doc = lemma(query)
        lemmatized_words = [token.lemma_ for token in doc]
        return lemmatized_words

    def nGrams(self, query):
        unigram_list = list(ngrams(query.split(), 1))
        bigram_list = list(ngrams(query.split(), 2))
        return unigram_list, bigram_list

    def partOfSpeech(self, query):
        nlp = self.nlp
        doc = nlp(query)
        part_of_speech = [(token.text, token.pos_) for token in doc]
        return part_of_speech

    def dependencyParsing(self, query):
        nlp = self.nlp
        doc = nlp(query)
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        return dependencies

    def keywordExtraction(self, query):
        vectorizer = TfidfVectorizer(max_features=10)
        matrix = vectorizer.fit_transform(query.split())
        keywords = vectorizer.get_feature_names_out()
        return keywords, matrix