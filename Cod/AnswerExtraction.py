import spacy
from nltk.corpus import stopwords
import regex as re
from nltk.stem import PorterStemmer

nlp = spacy.load('en_core_web_sm')
merge_chunk_flag = False

mapping = {
    'abb': None,
    'exp': None,
    'animal': 'LOC',
    'body': 'LOC',
    'color': 'ENTY',
    'cremat': 'ORG',
    'currency': 'ENTY',
    'dismed': 'ENTY',
    'event': 'EVENT',
    'food': 'PRODUCT',
    'instru': 'PRODUCT',
    'lang': 'LANGUAGE',
    'letter': 'ENTY',
    'other': 'ENTY',
    'plant': 'PRODUCT',
    'product': 'PRODUCT',
    'religion': 'ENTY',
    'sport': 'ENTY',
    'substance': 'ENTY',
    'symbol': 'ENTY',
    'techmeth': 'ENTY',
    'termeq': 'ENTY',
    'veh': 'PRODUCT',
    'word': 'ENTY',
    'def': None,
    'desc': None,
    'manner': None,
    'reason': None,
    'gr': 'ORG',
    'ind': 'PERSON',
    'title': 'PERSON',
    'city': 'GPE',
    'country': 'GPE',
    'mount': 'LOC',
    'state': 'GPE',
    'code': None,
    'count': 'CARDINAL',
    'date': 'DATE',
    'money': 'MONEY',
    'perc': 'PERCENT',
    'weight': 'QUANTITY'
}

class AnswerExtraction:
    def __init__(self):
        self.m_all_sentences = []
        self.m_sentences_separators = []
        self.m_sentences_after_separators = []
        self.m_token_lists_keywords = []
        self.m_matching_sequences_list = []
        self.m_combined_scored = []

    def rootForm(self, word):
        stemmer = PorterStemmer()
        lematizare = nlp(word)[0].lemma_
        return stemmer.stem(lematizare)

    def findRelevantSentences(self, keywords, document):
        separators = ['`', '~', '_', '\\', '<', '>', '=', '.', ' ', '+', ',', '?', '!', '"', '(', ')', '{', '}', '[',
                             ']', '\t', '$', ':', ';', '#', '@', '%', '|', '\b', '/', '*', '^', '-', '\r', '\v', '\f', '&',
                             '\'', '\n']
        stopWords = list(set(stopwords.words('english')))

        # Citesc toate propozitiile
        path = r'D:\Facultate\MASTER\Anul 2\Disertatie\Cod\Dataset\text_data\\' + document
        with open(path, 'r', encoding='latin1') as text_file:
            doc = nlp(text_file.read())
        self.m_all_sentences = [sent for sent in doc.sents]
        
        temp_list = []
        for sentence in self.m_all_sentences:
            if len(sentence) > 2:
                temp_list.append(sentence)

        self.m_all_sentences = temp_list
        self.m_all_sentences = list(enumerate(self.m_all_sentences))

        # Elimin separatori
        for sentence in self.m_all_sentences:
            if len(sentence[1]) > 2:
                pattern = '|'.join(map(re.escape, separators))
                temp = re.split(pattern, sentence[1].text)
                self.m_sentences_separators.append((sentence[0], [word.lower() for word in temp if word]))

        
        for sentence in self.m_sentences_separators:
            lista = []
            if len(sentence[1]) > 2:
                for word in sentence[1]:
                    if len(word) > 1:
                        lista.append(word)
                self.m_sentences_after_separators.append((sentence[0], lista))

        temp2_list = []
        temp3_list = []
        for sentence in self.m_sentences_after_separators:
            temp2_list.append(sentence[1])
        temp3_list = self.m_sentences_after_separators
        self.m_sentences_after_separators = temp2_list

        # Elimin stop-words
        for sentence in self.m_sentences_after_separators:
            stopwords_list = []
            for word in sentence:
                if word not in stopWords:
                    stopwords_list.append(word)

            token_lists = [nlp(word)[0] for word in stopwords_list]
            self.m_sentences_after_separators[self.m_sentences_after_separators.index(sentence)] = token_lists


        # Propozitiile sunt lematizate
        for sentence in self.m_sentences_after_separators:
            lemma_list = []
            for token in sentence:
                lemma_list.append(self.rootForm(token.text))
            self.m_sentences_after_separators[self.m_sentences_after_separators.index(sentence)] = lemma_list

        keywords_list = []
        for keyword in keywords:
            temp = keyword.split()
            for elem in temp:
                keywords_list.append(elem.lower())
        self.m_token_lists_keywords = [nlp(word)[0] for word in keywords_list]

        for i in range(len(self.m_token_lists_keywords)):
            self.m_token_lists_keywords[i] = self.rootForm(self.m_token_lists_keywords[i].text)

        # Parcurg fiecare cuvant dintr-o propozitie si verific cu secventa de keywords
        # Same word sequence score
        counter = 0
        for sentence_tokens in self.m_sentences_after_separators:
            matching_sequences_count = 0
            j = 0
            while j < len(sentence_tokens) - 1:
                copy_token_list_keywords = self.m_token_lists_keywords.copy()
                # Daca cuvant este in keywords
                if sentence_tokens[j] in self.m_token_lists_keywords:
                    temp_counter = 0
                    k = 0
                    # Parcurg cuvantul urmator pana la len(keywords)
                    while k < len(self.m_token_lists_keywords):
                        if j + k < len(sentence_tokens) and sentence_tokens[j + k] in copy_token_list_keywords:
                            temp_counter += 1
                            copy_token_list_keywords.remove(sentence_tokens[j + k])
                            k += 1
                        elif j + k >= len(sentence_tokens) or sentence_tokens[j + k] not in copy_token_list_keywords:
                            j += k-1
                            break
                    if temp_counter > matching_sequences_count:
                        matching_sequences_count = temp_counter
                j += 1


            self.m_matching_sequences_list.append([counter, matching_sequences_count])
            counter += 1

        keywords_score = []
        counter = 0
        for sentence in self.m_sentences_after_separators:
            copy_token_list_keywords = self.m_token_lists_keywords.copy()
            found_counter = 0
            for word in sentence:
                if word in copy_token_list_keywords:
                    copy_token_list_keywords.remove(word)
                    found_counter += 1
            keywords_score.append([counter, found_counter])
            counter += 1

        for i in range(0, len(self.m_matching_sequences_list)):
            self.m_combined_scored.append([i, self.m_matching_sequences_list[i][1] + keywords_score[i][1]])

        self.m_combined_scored = sorted(self.m_combined_scored, key=lambda x: -x[1])
        return temp3_list

    def namedEntityRecognition(self, list_of_possbile_answers, prediction_result, keywords):
        returned_values = []
        entity_found = False
        dict_of_entities = {}
        for sentence in list_of_possbile_answers:
            dict_of_entities[sentence[1]] = {}
            for elem in sentence[1].ents:
                entity_found = True
                dict_of_entities[sentence[1]][elem.text] = elem.label_
        if entity_found:
            dict_of_entities_copy = {}
            linked_result = mapping[prediction_result[0]]
            for sentence in dict_of_entities:
                has_prediction = False
                key_list = []
                for word in dict_of_entities[sentence]:
                    if(dict_of_entities[sentence][word]) is linked_result:
                        has_prediction = True
                        key_list.append(word)
                if has_prediction:
                    dict_of_entities_copy[sentence] = {}
                    for item in key_list:
                        dict_of_entities_copy[sentence][item] = dict_of_entities[sentence][item]

            dict_dependency_entities = {}
            for sentence in dict_of_entities_copy:
                dict_dependency_entities[sentence] = {}
                for word in dict_of_entities_copy[sentence]:
                    dict_dependency_entities[sentence][word] = 0

            dict_dependency_entities = {}
            for sentence in dict_of_entities_copy:
                dict_dependency_entities[sentence] = {}
                for word in dict_of_entities_copy[sentence]:
                    dict_dependency_entities[sentence][word] = 0

            if len(list(dict_dependency_entities.keys())) != 0:
                sentence = list(dict_dependency_entities.keys())[0]
                global merge_chunk_flag
                if not merge_chunk_flag:
                    nlp.add_pipe("merge_noun_chunks")
                    merge_chunk_flag = True
                for sentence in dict_dependency_entities:
                    doc = nlp(sentence.text)
                    for elem in dict_dependency_entities[sentence]:
                        for token in doc:
                            if elem == token.text:
                                while (token.head):
                                    if any(s in token.head.text for s in keywords):
                                        dict_dependency_entities[sentence][elem] += 1
                                        break
                                    else:
                                        if token == token.head:
                                            break
                                        token = token.head
                max_score = 0
                max_key = None
                for outer_key, inner_dict in dict_dependency_entities.items():
                    for inner_key, inner_value in inner_dict.items():
                        if inner_value > max_score:
                            max_key = inner_key
                            max_score = inner_value
                if max_score > 0:
                    returned_values.append(max_key)
                    returned_values.append(list_of_possbile_answers[0][1])
                else:
                    returned_values.append(list_of_possbile_answers[0][1])
            else:
                returned_values.append(list_of_possbile_answers[0][1])
        else:
            returned_values.append(list_of_possbile_answers[0][1])
        return returned_values

