import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_STOP_WORDS = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()


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
