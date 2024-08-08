import re
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# Initialize stopwords and lemmatizer
stop = stopwords.words('english')
wl = WordNetLemmatizer()

# Mapping for expanding contractions
mapping = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", 
    "'cause": "because", "could've": "could have", "couldn't": "could not", 
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will", 
    "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
    "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
    "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would", 
    "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am", 
    "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
    "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", 
    "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", 
    "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", 
    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", 
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", 
    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", 
    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", 
    "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would", 
    "that'd've": "that would have", "that's": "that is", "there'd": "there would", 
    "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would", 
    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", 
    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", 
    "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
    "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", 
    "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", 
    "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", 
    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", 
    "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", 
    "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", 
    "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", 
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", 
    "you're": "you are", "you've": "you have"
}

def clean_text(text, lemmatize=True):
    """
    Clean and preprocess text for LDA.

    Parameters:
    text (str): The text to be cleaned.
    lemmatize (bool): Whether to lemmatize the text or not.

    Returns:
    str: The cleaned text.
    """
    # Remove HTML tags
    soup = BeautifulSoup(text, "html5lib")
    text = soup.get_text()

    # Expand contractions
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    # Remove emojis
    emoji_clean = re.compile("["
                             u"\U0001F600-\U0001F64F"  # emoticons
                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags=re.UNICODE)
    text = emoji_clean.sub(r'', text)

    # Add space after full stop
    text = re.sub(r'\.(?=\S)', '. ', text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove punctuation and convert to lowercase
    text = "".join([word.lower() for word in text if word not in string.punctuation])

    # Lemmatize and remove stopwords
    if lemmatize:
        text = " ".join([wl.lemmatize(word) for word in text.split() if word not in stop and word.isalpha()])
    else:
        text = " ".join([word for word in text.split() if word not in stop and word.isalpha()])

    return text

def transform_dl_fct(desc_text):
    """
    Prepare text for Deep Learning models (e.g., USE, BERT).
    """
    word_tokens = word_tokenize(desc_text)
    lw = [w.lower() for w in word_tokens if (not w.startswith("@")) and (not w.startswith("http"))]
    transf_desc_text = ' '.join(lw)
    return transf_desc_text
