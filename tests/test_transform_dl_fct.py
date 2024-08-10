# tests/test_transform_dl_fct.py
import pytest
from utils import transform_dl_fct
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

def test_transform_dl_fct_basic():
    input_text = "This is a test"
    expected_output = "this is a test"
    assert transform_dl_fct(input_text) == expected_output

def test_transform_dl_fct_tokenization():
    input_text = "Simple test case"
    expected_output = "simple test case"
    assert transform_dl_fct(input_text) == expected_output