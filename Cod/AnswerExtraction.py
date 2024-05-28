import math

import spacy
from nltk.corpus import stopwords
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load('en_core_web_sm')




def findRelevantSentences(keywords, document):
    separators = ['`', '~', '_', '\\', '<', '>', '=', '.', ' ', '+', ',', '?', '!', '"', '(', ')', '{', '}', '[',
                         ']', '\t', '$', ':', ';', '#', '@', '%', '|', '\b', '/', '*', '^', '-', '\r', '\v', '\f', '&',
                         '\'', '\n']
    stopWords = list(set(stopwords.words('english')))

    # Citesc toate propozitiile
    path = r'D:\Facultate\MASTER\Anul 2\Disertatie\Cod\Dataset\text_data\\' + document + '.txt.clean'
    with open(path, 'r', encoding='latin1') as text_file:
        doc = nlp(text_file.read())
    all_sentences = [sent for sent in doc.sents]
    sentences_separators = []

    temp_list = []
    for sentence in all_sentences:
        if len(sentence) > 2:
            temp_list.append(sentence)

    all_sentences = temp_list

    # Elimin separatori
    for sentence in all_sentences:
        if len(sentence) > 2:
            pattern = '|'.join(map(re.escape, separators))
            temp = re.split(pattern, sentence.text)
            sentences_separators.append([word.lower() for word in temp if word])

    # Elimin stop-words
    for sentence in sentences_separators:
        stopwords_list = []
        for word in sentence:
            if word not in stopWords:
                stopwords_list.append(word)

        token_lists = [nlp(word)[0] for word in stopwords_list]
        sentences_separators[sentences_separators.index(sentence)] = token_lists

    # Propozitiile sunt lematizate
    for sentence in sentences_separators:
        lemma_list = []
        for token in sentence:
            lemma_list.append(token.lemma_)
        sentences_separators[sentences_separators.index(sentence)] = lemma_list

    # Lista de unique words
    unique_words_list = []
    for sentence in sentences_separators:
        for word in sentence:
            if word not in unique_words_list:
                unique_words_list.append(word)

    # Matrice de frecveta pt fiecare propozitie
    list_of_frequency = []
    for sentence in sentences_separators:
        word_frequency_sentence = {elem: 0 for elem in unique_words_list}
        for word in sentence:
            if word in word_frequency_sentence:
                word_frequency_sentence[word] += 1
        list_of_frequency.append(word_frequency_sentence)

    word_frequency_keywords = {elem: 0 for elem in unique_words_list}

    # Cazul in care sunt mai multe cuvinte intr-un string ("the similarities") + elimin stopwords
    keywords_list = []
    for keyword in keywords:
        temp = keyword.split()
        for elem in temp:
            if elem not in stopWords:
                keywords_list.append(elem)

    # Matrice frecventa pentru keywords
    token_lists_keywords = [nlp(word)[0] for word in keywords_list]
    for word in token_lists_keywords:
        if word.lemma_ in unique_words_list:
            word_frequency_keywords[word.lemma_] += 1

    temp_interogation = max(word_frequency_keywords, key=word_frequency_keywords.get)
    maximum_interogation = word_frequency_keywords[temp_interogation]
    sum_words = 0

    # Normalizare keyword
    for word in word_frequency_keywords:
        word_frequency_keywords[word] = word_frequency_keywords[word] / maximum_interogation

    # Normalizare propozitii
    for doc in list_of_frequency:
        temp_doc = max(list_of_frequency[list_of_frequency.index(doc)], key=list_of_frequency[list_of_frequency.index(doc)].get)
        maximum_doc = list_of_frequency[list_of_frequency.index(doc)][temp_doc]
        for word in list_of_frequency[list_of_frequency.index(doc)]:
            list_of_frequency[list_of_frequency.index(doc)][word] = list_of_frequency[list_of_frequency.index(doc)][word] / maximum_doc

    list_of_similarity = []
    # Calcul similaritati
    for doc in list_of_frequency:
        for word in list_of_frequency[list_of_frequency.index(doc)]:
            sum_words += (list_of_frequency[list_of_frequency.index(doc)][word] - word_frequency_keywords[word]) ** 2
        list_of_similarity.append((doc, math.sqrt(sum_words)))
        sum_words = 0

    list_of_similarity.sort(key=lambda x: x[1])
    for i in range(10):
        print(all_sentences[list_of_frequency.index(list_of_similarity[i][0])])
    pass





findRelevantSentences(['The sociolinguistic situation', 'Arabic', 'a prime example', 'provides'], 'S09_set5_a3')


