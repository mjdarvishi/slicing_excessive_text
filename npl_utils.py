from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from config import init,standard_size
import math
from nltk.corpus import wordnet
from math import sqrt
from collections import defaultdict
init()


def measure_length(text):
    # Measure the length of the document in terms of tokens
    tokens = word_tokenize(text)
    return len(tokens)

def word_tokenizing_without_lemmatization(text):
    # Measure the length of the document in terms of tokens
    tokens = word_tokenize(text)
    return tokens


def word_tokenizing_with_lemmatization(text):
    # Tokenization, stopword removal, and lemmatization
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    lemmatizer = nltk.stem.WordNetLemmatizer()

    tokens = tokenizer.tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return filtered_tokens


def check_size(sentence, word_slice):
    combined_tokens_length = len(word_tokenize(' '.join(word_slice))) + len(word_tokenize(sentence))
    return combined_tokens_length <= standard_size


def mesuare_word_similarity(slice1,slice2):
    words1 = set(slice1.split())
    words2 = set(slice2.split())
    
    wordnet_similarity = 0.0
    for word1 in words1:
        for word2 in words2:
            synsets1 = wordnet.synsets(word1)
            synsets2 = wordnet.synsets(word2)
            for synset1 in synsets1:
                for synset2 in synsets2:
                    similarity = synset1.path_similarity(synset2)
                    if similarity is not None and similarity > wordnet_similarity:
                        wordnet_similarity = similarity
    return wordnet_similarity

def check_slices_similarity(slice1, slice2, threshold=0.8,algoritm=1):
    # Preprocess slices
    # Vectorization and calculation of cosine distance
    slice1=''.join(word_tokenizing_with_lemmatization(slice1))
    slice2=''.join(word_tokenizing_with_lemmatization(slice2))
    if algoritm==2:
        word_similarity=mesuare_word_similarity(slice1,slice2)
        return (1 - word_similarity)>=threshold
    vectorizer =  CountVectorizer() if algoritm==0 else TfidfVectorizer()
    vectors = vectorizer.fit_transform([slice1, slice2])
    cosine_distance = cosine_distances(vectors)
    
    cosine_distance_score = cosine_distance[0, 1]
    return cosine_distance_score >= threshold
    
    

def split_into_slices(input_text):
    slices= []
    current_slice = []
    adjacent_slice = []
    sentences= sent_tokenize(input_text)
    for sentence in sentences:
        if len(current_slice) == 0:
            current_slice.append(sentence)
            continue
        if len(adjacent_slice) > 0 and check_slices_similarity(' '.join(adjacent_slice),' '.join(current_slice)):
            while not check_size(sentence, current_slice):
                current_slice.pop(0)

            current_slice.append(sentence)
            continue

        slice_token_size = len(word_tokenize(' '.join(current_slice)))
        sent_token_size = len(word_tokenize(sentence))

        if slice_token_size + sent_token_size > standard_size:
            slices.append(' '.join(current_slice))
            adjacent_slice = current_slice.copy()

            while not check_size(sentence, current_slice):
                current_slice.pop(0)

            current_slice.append(sentence)
        else:
            current_slice.append(sentence)
    print(slices)
    while check_slices_similarity(slices[-1], ' '.join(current_slice)):
        current_slice.pop(0)

        if len(current_slice) == 0:
            break

    if len(current_slice) != 0:
        slices.append(' '.join(current_slice))

    return slices