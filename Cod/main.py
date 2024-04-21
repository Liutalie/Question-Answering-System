import TextProcessing

if __name__ == '__main__':
    textProcessing = TextProcessing.TextProcessing()
    textProcessing.readStopWords()
    # textProcessing.translateText()
    query = "Acesta este un exemplu de query"
    query = query.lower()

    queryStemming = textProcessing.stemmingOfWords(query)
    print(queryStemming)

    queryLemma = textProcessing.lemmaOfWords(query)
    print(queryLemma)

    queryStopWords = textProcessing.removeStopWords(query)
    print(queryStopWords)

    queryUnigram, queryBigram = textProcessing.nGrams(query)
    print(queryUnigram)
    print(queryBigram)

    part_of_speech = textProcessing.partOfSpeech(query)
    print(part_of_speech)

    dependencies = textProcessing.dependencyParsing(query)
    print(dependencies)

    keywords, matrix = textProcessing.keywordExtraction(query)
    print(matrix)
    print(keywords)
