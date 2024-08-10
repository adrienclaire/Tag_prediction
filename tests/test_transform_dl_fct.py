# tests/test_transform_dl_fct.py
import pytest
from utils import transform_dl_fct
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def test_transform_dl_fct_basic():
    input_text = "This is a test."
    expected_output = "test"
    assert "test" in transform_dl_fct(input_text)

def test_transform_dl_fct_with_symbols():
    input_text = "Test - with / symbols #!"
    expected_output = "test symbols"
    assert transform_dl_fct(input_text) == expected_output

def test_transform_dl_fct_with_links():
    input_text = "Check this link: http://example.com"
    expected_output = "check link"
    assert transform_dl_fct(input_text) == expected_output
