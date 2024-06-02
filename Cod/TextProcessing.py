from nltk.stem.snowball import SnowballStemmer
import spacy
from nltk.corpus import wordnet
import regex as re

merge_chunks = False


class TextProcessing:
    def __init__(self):
        self.m_stopWords = []
        self.nlp = spacy.load('en_core_web_sm')
        self.datasets = {}
        self.stopwords = []

    def readData(self):
        trainData = {}
        testData = {}

        f1 = open("TREC/train_data.txt")
        f2 = open("TREC/test_data.txt")

        for line in f1:
            firstSplit = line.split(" ", 1)
            classes = firstSplit[0]
            classesSplit = classes.split(":")
            question = firstSplit[1]
            trainData[question.rstrip("\n")] = classesSplit

        for line in f2:
            classes = line.split(" ", 1)[0]
            classesCourse = classes.split(":")[0]
            classesFine = classes.split(":")[1]
            question = line.split(" ", 1)[1]
            testData[question.rstrip("\n")] = [classesCourse, classesFine]

        self.datasets['train'] = trainData
        self.datasets['test'] = testData

    def extractHeadWord(self, query):
        listOfDependencies = {}
        headword = ''
        for token in query:
            # build the dependency list
            listOfDependencies[token.dep_] = token

        if ("nsubj" in listOfDependencies) or (
                "nsubjpass" in listOfDependencies):  # if the question has a subject
            for token in query:
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    if token.pos_ in ["NOUN", "ADV",
                                      "PROPN"]:  # if the subj is a noun or adverb, pick it as the headword
                        headword = token.lemma_.lower()
                        break
                    if token.pos_ == "PRON":
                        for token2 in query:
                            if token2.dep_ == "dobj" and token2.pos_ in ["NOUN",
                                                                         "ADV"]:  # if the subj is a pronoun, get the direct object
                                headword = token2.lemma_.lower()
                                break
                            if token2.dep_ == "dobj" and token2.pos_ == "PROPN":
                                headword = "name"
        elif "attr" in listOfDependencies:
            for token in query:
                if token.dep_ == "attr" and (token.pos_ in ["NOUN", "ADV", "PROPN"]):
                    headword = token.lemma_.lower()

        else:  # if the question has no subject, consider it to be a "Name x" type of question and get the direct obj
            for token in query:
                if token.dep_ == "dobj" and token.pos_ in ["NOUN", "ADV"]:
                    headword = token.lemma_.lower()
                    break

        if headword == "":  # if no headword has been found, pick the first noun as the headword
            for token in query:
                if token.pos_ in ["NOUN", "PROPN"]:
                    headword = token.lemma_.lower()
                    break

        return headword

    def extractHypernim(self, headword):
        hypernim = ''
        if len(wordnet.synsets(headword)) > 0:
            syn = wordnet.synsets(headword)[0]
            hypernim = syn.hypernym_paths()[0]
        return hypernim

    def hypernimUntilRoot(self, hypernim):
        if len(hypernim) > 6:
            hypernym = hypernim[6]  # to be checked
            hypernym = str(hypernym).split(".")[0]
            hypernym = hypernym.split("'")[1]
        else:
            hypernym = hypernim[len(hypernim) - 1]  # to be checked
            hypernym = str(hypernym).split(".")[0]
            hypernym = hypernym.split("'")[1]
        return hypernym

    def queryExpansion(self, hypernim):
        queryExpansion = []
        if len(hypernim) > 0:
            hypWeight = 1
            for hyp in reversed(hypernim):
                hyp = str(hyp).split(".")[0]
                hyp = hyp.split("'")[1]
                queryExpansion.append(hyp + "~" + str(hypWeight))
                hypWeight = round(hypWeight * 60 / 100, 2)
        return queryExpansion

    def readStopwords(self):
        with open('stopwords.txt', 'r') as f:
            self.stopwords = f.read().splitlines()

    def removeStopWords(self, query):
        words = query.split()
        newText = []
        for word in words:
            if word.lower() not in self.m_stopWords:
                newText.append(word.lower())
        return newText

    def nGrams(self, query):
        unigram_list = [x.text for x in query]
        bigram_list = []
        for i in range(len(query)-1):
            bigram_list.append(query[i].lemma_.lower() + "-" + query[i+1].lemma_.lower())
        return unigram_list, bigram_list

    def extractKeywords(self, question):
        self.readStopwords()
        keywords = []

        # prepare the question
        question = question.replace("''", "\"")
        question = question.replace("``", "\"")

        # heuristic 1

        quotedWords = re.findall('"([^"]*)"', question)
        for quote in quotedWords:
            temp = self.nlp(quote.rstrip().lstrip())

            for token in temp:
                if token.text.lower() not in self.stopwords:
                    keywords.append(token.text.lower())

        # heuristics
        self.nlp.add_pipe("merge_noun_chunks")
        doc = self.nlp(question)
        for item in doc:
            if item.pos_ in ["NOUN", "PROPN"]:
                keywords.append(item.text)

        self.nlp.remove_pipe("merge_noun_chunks")

        # heuristic 5

        for token in doc:
            if token.pos_ == "VERB":
                keywords.append(token.text)

        return keywords