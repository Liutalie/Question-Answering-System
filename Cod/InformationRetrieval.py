import nltk
from nltk.corpus import stopwords

class InformationRetrieval:
    def __init__(self):
        self.m_stopWords = list(set(stopwords.words('english')))
        self.m_abbreviations = {}
        self.m_separators = [' ', '+', ',', '?', '!', '"', '(', ')', '{', '}', '[', ']', '\t', '$', ':', ';', '#', '@', '%', '|', '\b', '/', '*', '^', '-', '\r', '\v', '\f', '&', '\'']

        with open(r'D:\Facultate\MASTER\Anul 2\Disertatie\Cod\Dataset\abrevieri.txt') as folder:
            #PRIMUL CUVANT SE VEDE URAT
            abb_list = [x.split('\n')[0] for x in folder.readlines()]
            for x in range(0,len(abb_list),2):
                self.m_abbreviations[abb_list[x]] = abb_list[x+1]