import pandas
import sklearn
import numpy as np
import pickle
from sklearn.svm import LinearSVC

import TextProcessing

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
                                                                                    shuffle=False)

    global tempX
    global tempY
    tempX = X_test1
    tempY = y_test1
    myDataframe = X_train1
    true_positive_training = y_train1
    SVM.fit(myDataframe, true_positive_training)
    with open('SVM_Model.pkl','wb') as f:
        pickle.dump(SVM,f)

    # with open('SVM_Model.pkl', 'rb') as f:
    #     SVM = pickle.load(f)


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


if __name__ == '__main__':
    textProcessing = TextProcessing.TextProcessing()
    textProcessing.readData()

    training_feature_list = []
    testing_feature_list = []
    true_positive_training = []
    true_positive_testing = []
    SVM = LinearSVC()


    for dataset in textProcessing.datasets.items():
        for key in dataset[1]:
            question = key
            category = dataset[1][key][1]

            doc = textProcessing.nlp(question)
            # question features
            unigrams = []
            bigrams = []
            wordShapes = []
            headword = ''
            hypernym = '-'
            relatedWords = []
            queryExpansion = []
            questionCategory = ''

            wordShapeDict = {"lower": 0, "mixed": 0, "numbers": 0, 'other': 0}

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

    trainAlgorithm(training_feature_list,true_positive_training,SVM)
    testAlgorithm(testing_feature_list,true_positive_testing,SVM)




    # textProcessing.readStopWords()
    # # textProcessing.translateText()
    # query = "Acesta este un exemplu de query"
    # query = query.lower()
    #
    # queryStemming = textProcessing.stemmingOfWords(query)
    # print(queryStemming)
    #
    # queryLemma = textProcessing.lemmaOfWords(query)
    # print(queryLemma)
    #
    # queryStopWords = textProcessing.removeStopWords(query)
    # print(queryStopWords)
    #
    # queryUnigram, queryBigram = textProcessing.nGrams(query)
    # print(queryUnigram)
    # print(queryBigram)
    #
    # part_of_speech = textProcessing.partOfSpeech(query)
    # print(part_of_speech)
    #
    # dependencies = textProcessing.dependencyParsing(query)
    # print(dependencies)
    #
    # keywords, matrix = textProcessing.keywordExtraction(query)
    # print(matrix)
    # print(keywords)
