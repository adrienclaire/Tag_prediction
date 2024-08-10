# tests/test_transform_dl_fct.py
import pytest
from utils import transform_dl_fct
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

def test_transform_dl_fct_basic():
    input_text = "This is a test."
    expected_output = "this is a test"
    assert transform_dl_fct(input_text) == expected_output

def test_transform_dl_fct_with_mentions_and_urls():
    input_text = "Check out this link http://example.com and mention @user"
    expected_output = "check out this link and mention"
    assert transform_dl_fct(input_text) == expected_output

def test_transform_dl_fct_with_symbols():
    input_text = "Test - with / symbols #!"
    expected_output = "test - with / symbols # !"
    assert transform_dl_fct(input_text) == expected_output

def test_transform_dl_fct_with_mixed_case():
    input_text = "This Is A Test Of Mixed CASE."
    expected_output = "this is a test of mixed case"
    assert transform_dl_fct(input_text) == expected_output