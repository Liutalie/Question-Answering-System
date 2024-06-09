import os
import regex as re
import spacy
from nltk.corpus import stopwords

import AnswerExtraction
import pandas
import sklearn
import numpy as np
import pickle
from tkinter import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import GUI
import TextProcessing, InformationRetrieval

nlp = spacy.load('en_core_web_sm')

tempX = None
tempY = None
row_template = None


def trainAlgorithm(training_feature_list, true_positive_training, SVM):
    global row_template
    new_matrix = []
    all_features = []
    for elem_list in training_feature_list:
        for elem in elem_list:
            all_features.append(elem)

    unique_features = list(set(all_features))
    row_template = dict.fromkeys(unique_features, 0)

    for feature_list in training_feature_list:
        row = row_template.copy()
        for elem in feature_list:
            if '~' in elem:
                temp = elem.split('~')
                row[elem] = float(temp[1])
            else:
                row[elem] = 1

        new_matrix.append(row)
    data_frame = pandas.DataFrame(new_matrix)

    X_train1, X_test1, y_train1, y_test1 = sklearn.model_selection.train_test_split(data_frame,
                                                                                    true_positive_training, test_size=0.2,
                                                                                    shuffle=True, stratify=true_positive_training)

    global tempX
    global tempY
    tempX = X_test1
    tempY = y_test1
    data_frame = X_train1
    true_positive_training = y_train1
    SVM.fit(data_frame, true_positive_training)
    with open('SVM_Model_stratify2.pkl', 'wb') as f:
        pickle.dump(SVM, f)


def testAlgorithm(testing_feature_list, true_positive_testing, SVM):
    global tempX
    global tempY
    global row_template
    new_matrix = []

    for feature_list in testing_feature_list:
        row = row_template.copy()
        for elem in feature_list:
            if elem in list(row.keys()):
                row[elem] = 1

        new_matrix.append(row)
    data_frame = pandas.DataFrame(new_matrix)

    results = SVM.predict(data_frame)

    testData = sklearn.metrics.classification_report(results, true_positive_testing,
                                                              zero_division=0,labels=np.unique(results))
    print(testData)


def getTrainTestFeatures(textProcessing):
    training_feature_list = []
    testing_feature_list = []
    true_positive_training = []
    true_positive_testing = []
    SVM = LinearSVC()
    textProcessing.readData()
    for dataset in textProcessing.m_datasets.items():
        for key in dataset[1]:
            question = key
            category = dataset[1][key][1]
            doc = textProcessing.m_nlp(question)
            # question features
            unigrams = []
            bigrams = []
            headword = ''
            hypernym = '-'
            query_expansion = []
            unigrams, bigrams = textProcessing.nGrams(doc)
            headword = textProcessing.extractHeadWord(doc)
            if headword == '':
                headword = '-'
            else:
                hypernym = textProcessing.extractHypernim(headword)
                if len(hypernym) > 0:
                    query_expansion = textProcessing.queryExpansion(hypernym)
                    hypernym = textProcessing.hypernimUntilRoot(hypernym)
            list_of_feature = [unigrams, bigrams, headword, hypernym, query_expansion]
            unpacked_features = []
            for item in list_of_feature:
                if isinstance(item, list):
                    unpacked_features.extend(item)
                else:
                    unpacked_features.append(item)

            if dataset[0] == 'train':
                training_feature_list.append(unpacked_features)
                true_positive_training.append(category)
            else:
                testing_feature_list.append(unpacked_features)
                true_positive_testing.append(category)

    trainAlgorithm(training_feature_list, true_positive_training, SVM)
    testAlgorithm(testing_feature_list, true_positive_testing, SVM)


def predictQuestionCategory(question):
    with open('SVM_Model_stratify2.pkl', 'rb') as f:
        SVM = pickle.loads(f.read())
    textProcessing = TextProcessing.TextProcessing()
    textProcessing.readData()
    testing_feature_list = []
    doc = textProcessing.m_nlp(question)

    # question features
    unigrams = []
    bigrams = []
    headword = ''
    hypernym = '-'
    queryExpansion = []
    unigrams, bigrams = textProcessing.nGrams(doc)
    headword = textProcessing.extractHeadWord(doc)
    if headword == '':  # could not find the headword
        headword = '-'
    else:
        hypernym = textProcessing.extractHypernim(headword)
        if len(hypernym) > 0:
            queryExpansion = textProcessing.queryExpansion(hypernym)
            hypernym = textProcessing.hypernimUntilRoot(hypernym)
    list_of_feature = [unigrams, bigrams, headword, hypernym, queryExpansion]
    unpacked_features = []
    for item in list_of_feature:
        if isinstance(item, list):
            unpacked_features.extend(item)
        else:
            unpacked_features.append(item)
    testing_feature_list.append(unpacked_features)
    feature_names = list(SVM.feature_names_in_)
    question_frequency_dict = {elem: 0 for elem in feature_names}

    new_matrix = []
    for feature_list in testing_feature_list:
        for elem in feature_list:
            if elem in list(question_frequency_dict.keys()):
                question_frequency_dict[elem] = 1

        new_matrix.append(question_frequency_dict)
    data_frame = pandas.DataFrame(new_matrix)
    return (SVM.predict(data_frame))


def getAnswer(question, infoRetrieval, textProcessing, answerExtraction):
    prediction_result = predictQuestionCategory(question)

    infoRetrieval.documentProcessing()
    infoRetrieval.frequencyOfWords()
    infoRetrieval.questionProcessing(question)
    infoRetrieval.Normalization()

    keywords = textProcessing.extractKeywords(question)

    sentences_after_separators = answerExtraction.findRelevantSentences(
        keywords, infoRetrieval.m_list_of_similarity[0][0])

    list_of_possbile_answers = []
    temp_max = answerExtraction.m_combined_scored[0][1]
    for elem in answerExtraction.m_combined_scored:
        if elem[1] == temp_max:
            list_of_possbile_answers.append(elem[0])
        else:
            break

    for elem in list_of_possbile_answers:
        for elem2 in sentences_after_separators:
            if elem == sentences_after_separators.index(elem2):
                list_of_possbile_answers[list_of_possbile_answers.index(elem)] \
                    = answerExtraction.m_all_sentences[sentences_after_separators[sentences_after_separators.index(elem2)][0]]

    answer = answerExtraction.namedEntityRecognition(list_of_possbile_answers, prediction_result, keywords)
    return answer, infoRetrieval.m_list_of_similarity[0][0]


def getAccuracy():
    list_of_elimination = ['no', 'yes', 'null', 'yes.', 'no.', '']
    all_question = []
    path = r'D:\Facultate\MASTER\Anul 2\Disertatie\Cod\Dataset\Questions'
    for file in os.listdir(path):
        file_path = f"{path}\{file}"
        with open(file_path, 'r', encoding='latin1') as text_file:
            all_question.extend(text_file.readlines()[1:])
    all_question_split = []
    for elem in all_question:
        elem = elem.split('\t')
        if elem[2].lower() not in list_of_elimination and elem[1] not in list_of_elimination:
            all_question_split.append([elem[1], elem[2], elem[5].split('\n')[0]])

    unique_question_list = []
    for elem in all_question_split:
        if len(unique_question_list) > 0:
            if elem[0] == unique_question_list[-1][0]:
                pass
            else:
                unique_question_list.append(elem)
        else:
            unique_question_list.append(elem)

    counter_question = 1
    counter_info_retrieval = 0
    counter_answer = 0
    stopWords = list(set(stopwords.words('english')))
    separators = ['`', '~', '_', '\\', '<', '>', '=', '.', ' ', '+', ',', '?', '!', '"', '(', ')', '{', '}', '[',
                  ']', '\t', '$', ':', ';', '#', '@', '%', '|', '\b', '/', '*', '^', '-', '\r', '\v', '\f', '&',
                  '\'', '\n']
    pattern = '|'.join(map(re.escape, separators))
    for sentence in unique_question_list:
        stopwords_list = []
        temp = sentence[1].split(' ')
        for word in temp:
            if word not in stopWords:
                stopwords_list.append(word.lower())

        temp_sep = []
        for word in stopwords_list:
            temp_sep.extend(re.split(pattern, word))
        joined_list = ' '.join(temp_sep)
        doc = nlp(joined_list)
        token_list = [word for word in doc]
        for token in token_list:
            token_list[token_list.index(token)] = token.lemma_
        unique_question_list[unique_question_list.index(sentence)][1] = [word for word in token_list]

    for elem in unique_question_list:
        try:
            infoRetrieval = InformationRetrieval.InformationRetrieval()
            textProcessing = TextProcessing.TextProcessing()
            answerExtraction = AnswerExtraction.AnswerExtraction()
            answer, document = getAnswer(elem[0], infoRetrieval, textProcessing, answerExtraction)
            document = document.split('.txt')[0]
            temp_sep = []
            if len(answer) == 1:
                answer = ['', answer[0]]
            for word in answer:
                temp_sep.extend(re.split(pattern, str(word)))
            joined_list = ' '.join(temp_sep)
            doc = nlp(joined_list)
            token_list = [word for word in doc]
            for token in token_list:
                token_list[token_list.index(token)] = token.lemma_.lower()
            if all([item in token_list for item in elem[1]]):
                counter_answer += 1
            if elem[2] in document:
                counter_info_retrieval += 1
            print("Information retrieval accuracy: " + str((counter_info_retrieval / counter_question * 100)) + '%')
            print("Expected: " + elem[2] + "  Retrieved: " + document)
            print("Answer accuracy: " + str((counter_answer / counter_question * 100)) + "%")
            print("Expected: " + ' '.join(elem[1]))
            print("Retrieved: " + str(answer[0]))
            print("Retrieved: " + str(answer[1]))
            print("Question number: " + str(counter_question) + "\n")
            counter_question += 1
        except:
            print('-----------------------------------------')
            counter_question += 1
            print(elem)
            print('-----------------------------------------')

if __name__ == '__main__':
    root = Tk()
    root.title("Question Answering System")
    root.geometry("1200x500")
    gui = GUI.GUI(root)
    gui.pack(fill="both", expand=True)
    root.mainloop()

    # getAccuracy()

    # infoRetrieval = InformationRetrieval.InformationRetrieval()
    # textProcessing = TextProcessing.TextProcessing()
    # answerExtraction = AnswerExtraction.AnswerExtraction()
    # # getTrainTestFeatures(textProcessing)
    # getAnswer('Which countries established colonies in Canada?', infoRetrieval, textProcessing, answerExtraction)
