# tests/test_clean_text.py
import pytest
from utils import clean_text

def test_clean_text_basic():
    input_text = "<p>This is a <b>test</b>!</p>"
    expected_output = "test"
    assert clean_text(input_text) == expected_output

def test_clean_text_with_stopwords():
    input_text = "This is a very simple test."
    expected_output = "simple test"
    assert clean_text(input_text) == expected_output

def test_clean_text_with_emojis():
    input_text = "Test with emoji 😊!"
    expected_output = "test emoji"
    assert clean_text(input_text) == expected_output