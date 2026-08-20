import pytest

from src.preprocessing import (
    clean_html,
    eh_hinglish,
    expand_contractions,
    normalize_token,
    preprocess_pipeline,
    rating_to_sentiment,
)


def test_clean_html_strips_html_without_gluing_words():
    assert clean_html("ever<br/>No") == "ever No"


def test_clean_html_preserves_case_and_punctuation():
    assert clean_html("Bad!!! Service.") == "Bad!!! Service."


def test_clean_html_collapses_whitespace():
    assert clean_html("multiple   spaces   here") == "multiple spaces here"


def test_clean_html_decodes_html_entities_without_leaving_amp_token():
    assert clean_html("chicken &amp; rice") == "chicken & rice"


def test_normalize_token_lowercases_and_removes_punctuation():
    assert normalize_token("Bad!!!") == "bad"


def test_normalize_token_pure_punctuation_becomes_empty():
    assert normalize_token("!!!") == ""


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


def test_preprocess_pipeline_stopword_removal_does_not_leak_irregular_lemma():
    tokens = preprocess_pipeline("The food was great")
    assert "be" not in tokens


def test_preprocess_pipeline_lemmatizes_verb_using_pos_tag():
    tokens = preprocess_pipeline("I never received my order")
    assert "receive" in tokens
    assert "received" not in tokens


def test_preprocess_pipeline_decodes_html_entities_without_leaving_amp_token():
    tokens = preprocess_pipeline("chicken &amp; rice")
    assert "amp" not in tokens


def test_expand_contractions_irregular_forms():
    assert expand_contractions("won't") == "will not"
    assert expand_contractions("can't") == "can not"
    assert expand_contractions("shan't") == "shall not"


def test_expand_contractions_preserves_leading_case():
    assert expand_contractions("Won't") == "Will not"


def test_expand_contractions_generic_nt_suffix():
    assert expand_contractions("didn't") == "did not"
    assert expand_contractions("wasn't") == "was not"


def test_preprocess_pipeline_keeps_negation_as_own_token():
    tokens = preprocess_pipeline("The food didn't arrive on time")
    assert "not" in tokens
    assert "nt" not in tokens


def test_preprocess_pipeline_wont_expands_without_orphan_fragment():
    tokens = preprocess_pipeline("I won't order again")
    assert "not" in tokens
    assert "wo" not in tokens
