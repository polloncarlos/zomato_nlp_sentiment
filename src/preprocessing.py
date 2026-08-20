import html
import re
import nltk
from nltk.corpus import stopwords, words as nltk_words, wordnet
from nltk.stem import WordNetLemmatizer

_STOP_WORDS = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()
_ENGLISH_VOCAB = set(w.lower() for w in nltk_words.words())


def clean_html(text: str) -> str:
    """Decode HTML entities and strip HTML tags, preserving case and punctuation."""
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize_raw(text: str) -> list:
    """Tokenize text with case and punctuation intact, for POS tagging."""
    return nltk.word_tokenize(text)


def tag_pos(tokens: list) -> list:
    """Tag each token with its Penn Treebank part-of-speech, using sentence context."""
    return nltk.pos_tag(tokens)


def _wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank POS tag to the WordNet POS category (default: noun)."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def normalize_token(token: str) -> str:
    """Lowercase a token and strip non-alphabetic characters."""
    return re.sub(r'[^a-z]', '', token.lower())


def remove_stopwords(tagged_tokens: list) -> list:
    """Normalize (token, pos_tag) pairs and drop empties and NLTK English stopwords."""
    result = []
    for token, tag in tagged_tokens:
        normalized = normalize_token(token)
        if normalized and normalized not in _STOP_WORDS:
            result.append((normalized, tag))
    return result


def lemmatize_tagged(tagged_tokens: list) -> list:
    """Lemmatize (normalized_token, pos_tag) pairs using each token's own tag."""
    return [_lemmatizer.lemmatize(t, pos=_wordnet_pos(tag)) for t, tag in tagged_tokens]


def preprocess_pipeline(text: str) -> list:
    """Full pipeline: strip HTML → tokenize → POS tag → remove stopwords → lemmatize."""
    tokens = tokenize_raw(clean_html(text))
    tagged = tag_pos(tokens)
    filtered = remove_stopwords(tagged)
    return lemmatize_tagged(filtered)


def rating_to_sentiment(rating: int) -> str:
    """Map a 1-5 rating to a sentiment class (positivo/neutro/negativo)."""
    if rating >= 4:
        return 'positivo'
    if rating == 3:
        return 'neutro'
    return 'negativo'


def eh_hinglish(texto: str, limiar: float = 0.50) -> bool:
    """True if fewer than `limiar` of the text's tokens are recognized English words."""
    tokens = re.sub(r'[^a-z\s]', '', str(texto).lower()).split()
    if not tokens:
        return True
    pct_en = sum(1 for t in tokens if t in _ENGLISH_VOCAB) / len(tokens)
    return pct_en < limiar
