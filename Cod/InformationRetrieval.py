import math
import os
import regex as re
from nltk.corpus import stopwords


class InformationRetrieval:
    def __init__(self):
        self.m_stopWords = list(set(stopwords.words('english')))
        self.m_abbreviations = {}
        self.m_separators = ['`', '~', '_', '\\', '<', '>', '=', '.', ' ', '+', ',', '?', '!', '"', '(', ')', '{', '}', '[', ']', '\t', '$', ':', ';', '#', '@', '%', '|', '\b', '/', '*', '^', '-', '\r', '\v', '\f', '&', '\'', '\n']
        self.m_text_documents = {}
        self.m_unique_words = []
        self.m_word_frequency_all = {elem:None for elem in os.listdir(r'D:\Facultate\MASTER\Anul 2\Disertatie\Cod\Dataset\text_data') if elem.endswith('.clean')}
        self.m_word_frequency_interogation = {elem:None for elem in os.listdir(r'D:\Facultate\MASTER\Anul 2\Disertatie\Cod\Dataset\text_data') if elem.endswith('.clean')}
        self.m_list_of_similarity = []

        with open(r'D:\Facultate\MASTER\Anul 2\Disertatie\Cod\Dataset\abrevieri.txt') as folder:
            #PRIMUL CUVANT SE VEDE URAT
            abb_list = [x.split('\n')[0] for x in folder.readlines()]
            for x in range(0, len(abb_list), 2):
                self.m_abbreviations[abb_list[x].lower()] = abb_list[x+1]

    def documentProcessing(self):
        # Reading all documents
        path = r'D:\Facultate\MASTER\Anul 2\Disertatie\Cod\Dataset\text_data'
        for file in os.listdir(path):
            if file.endswith('.clean'):
                file_path = f"{path}\{file}"
                with open(file_path, 'r', encoding='latin1') as text_file:
                    self.m_text_documents[file] = text_file.read()

        # Procesarea documentelor
        for key in self.m_text_documents:
            # Elimin numerele
            self.m_text_documents[key] = re.sub(r'\d+', '', self.m_text_documents[key])

            # Elimin separatorii
            pattern = '|'.join(map(re.escape, self.m_separators))
            temp = re.split(pattern, self.m_text_documents[key])
            self.m_text_documents[key] = [word.lower() for word in temp if word]

            # Elimin stop words si abrevieri
            stopwords_list = []
            for word in self.m_text_documents[key]:
                if word in [abbrev for abbrev in list(self.m_abbreviations.keys())]:
                    word = self.m_abbreviations[word]

                if word not in self.m_stopWords:
                    stopwords_list.append(word)
            self.m_text_documents[key] = stopwords_list

            self.m_unique_words.extend(self.m_text_documents[key])

        # Cuvinte unice
        self.m_unique_words = list(set(self.m_unique_words))
        self.m_unique_words.sort()

    # Dictionar frecventa pt fiecare document
    def frequencyOfWords(self):
        for key in self.m_text_documents:
            word_frequency = {elem: 0 for elem in self.m_unique_words}
            for word in self.m_text_documents[key]:
                if word in word_frequency:
                    word_frequency[word] += 1
            self.m_word_frequency_all[key] = word_frequency

    def questionProcessing(self, question):
        question = re.sub(r'\d+', '', question)

        pattern = '|'.join(map(re.escape, self.m_separators))
        temp = re.split(pattern, question)
        question = [word.lower() for word in temp if word]

        stopwords_list = []
        for word in question:
            if word in [abbrev for abbrev in list(self.m_abbreviations.keys())]:
                word = self.m_abbreviations[word]

            if word not in self.m_stopWords:
                stopwords_list.append(word)
        question = stopwords_list
        question.sort()

        # Dictionar frecventa interogare
        word_frequency = {elem: 0 for elem in self.m_unique_words}
        for word in question:
            if word in word_frequency:
                word_frequency[word] += 1
        self.m_word_frequency_interogation = word_frequency

    def Normalization(self):
        temp_interogation = max(self.m_word_frequency_interogation, key=self.m_word_frequency_interogation.get)
        maximum_interogation = self.m_word_frequency_interogation[temp_interogation]
        sum_words = 0

        # Normalizare interogare
        for word in self.m_word_frequency_interogation:
            self.m_word_frequency_interogation[word] = self.m_word_frequency_interogation[word] / maximum_interogation

        # Normalizare documente
        for doc in self.m_word_frequency_all:
            temp_doc = max(self.m_word_frequency_all[doc], key=self.m_word_frequency_all[doc].get)
            maximum_doc = self.m_word_frequency_all[doc][temp_doc]
            for word in self.m_word_frequency_all[doc]:
                self.m_word_frequency_all[doc][word] = self.m_word_frequency_all[doc][word] / maximum_doc

        # Calcul similaritati
        for doc in self.m_word_frequency_all:
            for word in self.m_word_frequency_all[doc]:
                sum_words += (self.m_word_frequency_all[doc][word] - self.m_word_frequency_interogation[word]) ** 2
            self.m_list_of_similarity.append((doc, math.sqrt(sum_words)))
            sum_words = 0
        self.m_list_of_similarity.sort(key=lambda x: x[1])
