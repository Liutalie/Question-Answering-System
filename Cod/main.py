import AnswerExtraction
import pandas
import sklearn
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import TextProcessing, InformationRetrieval

tempX = None
tempY = None
rowTemplate = None


def trainAlgorithm(training_feature_list, true_positive_training, SVM):
    global rowTemplate
    newMatrix = []
    all_features = []
    for elem_list in training_feature_list:
        for elem in elem_list:
            all_features.append(elem)

    uniqueFeatures = list(set(all_features))
    rowTemplate = dict.fromkeys(uniqueFeatures, 0)

    for feature_list in training_feature_list:
        row = rowTemplate.copy()
        for elem in feature_list:
            row[elem] = 1

        newMatrix.append(row)
    myDataframe = pandas.DataFrame(newMatrix)

    X_train1, X_test1, y_train1, y_test1 = sklearn.model_selection.train_test_split(myDataframe,
                                                                                    true_positive_training, test_size=0.2,
                                                                                    shuffle=True, stratify=true_positive_training)

    global tempX
    global tempY
    tempX = X_test1
    tempY = y_test1
    myDataframe = X_train1
    true_positive_training = y_train1
    SVM.fit(myDataframe, true_positive_training)
    with open('SVM_Model_stratify.pkl', 'wb') as f:
        pickle.dump(SVM, f)


def testAlgorithm(testing_feature_list, true_positive_testing, SVM):
    global tempX
    global tempY
    global rowTemplate
    newMatrix = []

    for feature_list in testing_feature_list:
        row = rowTemplate.copy()
        for elem in feature_list:
            if elem in list(row.keys()):
                row[elem] = 1

        newMatrix.append(row)
    myDataframe = pandas.DataFrame(newMatrix)

    results = SVM.predict(myDataframe)

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
    for dataset in textProcessing.datasets.items():
        for key in dataset[1]:
            question = key
            category = dataset[1][key][1]
            doc = textProcessing.nlp(question)
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

            if dataset[0] == 'train':
                training_feature_list.append(unpacked_features)
                true_positive_training.append(category)
            else:
                testing_feature_list.append(unpacked_features)
                true_positive_testing.append(category)

    trainAlgorithm(training_feature_list, true_positive_training, SVM)
    testAlgorithm(testing_feature_list, true_positive_testing, SVM)


def predictQuestionCategory(question):
    with open('SVM_Model_stratify.pkl', 'rb') as f:
        SVM = pickle.loads(f.read())
    textProcessing = TextProcessing.TextProcessing()
    textProcessing.readData()
    testing_feature_list = []
    doc = textProcessing.nlp(question)

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

    newMatrix = []
    for feature_list in testing_feature_list:
        for elem in feature_list:
            if elem in list(question_frequency_dict.keys()):
                question_frequency_dict[elem] = 1

        newMatrix.append(question_frequency_dict)
    myDataframe = pandas.DataFrame(newMatrix)
    return (SVM.predict(myDataframe))


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
            if elem == elem2[0]:
                list_of_possbile_answers[list_of_possbile_answers.index(elem)] = answerExtraction.m_all_sentences[
                    sentences_after_separators[elem2[0]][0]]

    answerExtraction.namedEntityRecognition(list_of_possbile_answers, prediction_result, keywords)


if __name__ == '__main__':
    infoRetrieval = InformationRetrieval.InformationRetrieval()
    textProcessing = TextProcessing.TextProcessing()
    answerExtraction = AnswerExtraction.AnswerExtraction()
    # getTrainTestFeatures(textProcessing)
    getAnswer('Which countries established colonies in Canada?', infoRetrieval, textProcessing, answerExtraction)
