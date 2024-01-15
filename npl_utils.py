from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import nltk
from nltk.tokenize import word_tokenize
from config import init,standard_size
import math
from nltk.corpus import wordnet
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

    return " ".join(filtered_tokens)

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

def check_slices_similarity(slice1, slice2, threshold=0.2,algoritm=1):
    # Preprocess slices
    # Vectorization and calculation of cosine distance
    slice1=word_tokenizing_with_lemmatization(slice1)
    slice2=word_tokenizing_with_lemmatization(slice2)
    if algoritm==2:
        word_similarity=mesuare_word_similarity(slice1,slice2)
        return (1 - word_similarity)>=threshold
    vectorizer =  CountVectorizer() if algoritm==0 else TfidfVectorizer()
    vectors = vectorizer.fit_transform([slice1, slice2])
    cosine_distance = cosine_distances(vectors)
    
    cosine_distance_score = cosine_distance[0, 1]
    return cosine_distance_score >= threshold
    
def check_overlap(text1, text2):
    # Convert texts to lowercase for case-insensitive matching
    text1_lower =text1.lower()
    text2_lower = text2.lower()

    # Find all overlapping sequences of a certain length
    overlap_length = min(measure_length(text1), measure_length(text2))

    for length in range(overlap_length, 0, -1):
        for start in range(len(text1) - length + 1):
            if text1_lower[start:start + length] in text2_lower:
                return True

    return False

def split_into_slices(input_text):
    input_text=word_tokenizing_without_lemmatization(input_text)
    input_size = len(input_text)
    # Case 2: Input is above the standard size, split into slices
    num_slices = math.ceil(input_size / standard_size)
    slice_size = math.ceil(input_size / num_slices)

    slices = []
    for i in range(num_slices):
        start_index = i * slice_size
        end_index = min((i + 1) * slice_size, input_size)
        new_slice=input_text[start_index:end_index]
        new_slice_text=" ".join(new_slice)
        if i > 0:
            # get previose slice
            prev_slice = slices[-1]
            # checking overlap
            overlap = check_overlap(new_slice_text,prev_slice)
            # checking diffrence
            # Algoritms={0:'count_vector',1:'Tfidf_vectorizer',2:'meaning_similarity'}
            # default one is :1:'Tfidf_vectorizer.
            # becuase in the problem is mentioned to count of occurrences weighted on the length of the document 
            different_enough = check_slices_similarity(prev_slice,new_slice_text,algoritm=0)

            if not overlap or not different_enough:
                # Adjust the current slice to ensure no overlap and sufficient difference
                new_start_index = end_index - len(new_slice)
                new_slice = input_text[new_start_index:end_index]
                new_slice_text = " ".join(new_slice)

        slices.append(new_slice_text)

    return slices