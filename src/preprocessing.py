import re
from nltk.corpus import stopwords, words as nltk_words
from nltk.stem import WordNetLemmatizer

_STOP_WORDS = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()
_ENGLISH_VOCAB = set(w.lower() for w in nltk_words.words())


def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags, remove non-alpha characters, normalize whitespace."""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text: str) -> list:
    """Split cleaned text into tokens."""
    return text.split()


def remove_stopwords(tokens: list) -> list:
    """Remove NLTK English stopwords."""
    return [t for t in tokens if t not in _STOP_WORDS]


def lemmatize(tokens: list) -> list:
    """Lemmatize each token with WordNetLemmatizer."""
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess_pipeline(text: str) -> list:
    """Full pipeline: clean → tokenize → remove stopwords → lemmatize."""
    return lemmatize(remove_stopwords(tokenize(clean_text(text))))


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
