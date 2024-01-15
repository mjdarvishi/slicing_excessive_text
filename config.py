import nltk

# Example usage:
input_text = "This is an example text.An example is given here."

standard_size = 5
def init():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download("stopwords")
        nltk.download('wordnet')