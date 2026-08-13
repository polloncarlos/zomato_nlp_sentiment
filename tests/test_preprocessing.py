import pytest

from src.preprocessing import clean_text, eh_hinglish, preprocess_pipeline, rating_to_sentiment


def test_clean_text_strips_html_without_gluing_words():
    assert clean_text("ever<br/>No") == "ever no"


def test_clean_text_lowercases_and_removes_punctuation():
    assert clean_text("Bad!!! Service.") == "bad service"


def test_clean_text_collapses_whitespace():
    assert clean_text("multiple   spaces   here") == "multiple spaces here"


def test_clean_text_decodes_html_entities_without_leaving_amp_token():
    assert clean_text("chicken &amp; rice") == "chicken rice"


@pytest.mark.parametrize(
    "rating,esperado",
    [(1, "negativo"), (2, "negativo"), (3, "neutro"), (4, "positivo"), (5, "positivo")],
)
def test_rating_to_sentiment(rating, esperado):
    assert rating_to_sentiment(rating) == esperado


def test_eh_hinglish_texto_ingles_e_false():
    assert eh_hinglish("the food was great and tasty") is False


def test_eh_hinglish_texto_vazio_e_true():
    assert eh_hinglish("") is True


def test_preprocess_pipeline_remove_stopwords():
    tokens = preprocess_pipeline("The food was very tasty and I ordered again")
    assert "food" in tokens
    assert "tasty" in tokens
    assert "the" not in tokens
    assert "was" not in tokens
