import TextProcessing

if __name__ == '__main__':
    textProcessing = TextProcessing.TextProcessing()
    textProcessing.readStopWords()
    # textProcessing.translateText()
    query = "Ce vârstă are Messi"
    query = query.lower()

    queryStemming = textProcessing.stemmingOfWords(query)
    print(queryStemming)

    queryLemma = textProcessing.lemmaOfWords(query)
    print(queryLemma)

    queryStopWords = textProcessing.removeStopWords(query)
    print(queryStopWords)
