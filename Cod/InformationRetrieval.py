import os
import regex as re

import nltk
from nltk.corpus import stopwords

class InformationRetrieval:
    def __init__(self):
        self.m_stopWords = list(set(stopwords.words('english')))
        self.m_abbreviations = {}
        self.m_separators = ['`', '~', '_', '\\', '<', '>', '=', '.', ' ', '+', ',', '?', '!', '"', '(', ')', '{', '}', '[', ']', '\t', '$', ':', ';', '#', '@', '%', '|', '\b', '/', '*', '^', '-', '\r', '\v', '\f', '&', '\'', '\n']
        self.m_text_documents = {}
        self.m_unique_words = []
        self.m_word_frequency_all = {elem:None for elem in os.listdir(r'D:\Facultate\MASTER\Anul 2\Disertatie\Cod\Dataset\text_data') if elem.endswith('.clean')}


        with open(r'D:\Facultate\MASTER\Anul 2\Disertatie\Cod\Dataset\abrevieri.txt') as folder:
            #PRIMUL CUVANT SE VEDE URAT
            abb_list = [x.split('\n')[0] for x in folder.readlines()]
            for x in range(0,len(abb_list),2):
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
        pass








