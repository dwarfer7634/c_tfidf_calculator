from enum import Enum

from pydantic import BaseModel
from sudachipy import tokenizer
from sudachipy import dictionary
from sklearn.feature_extraction.text import TfidfVectorizer

class Mode(Enum):
    ONE_VS_ONE = "one_vs_one"
    ONE_VS_REST = "one_vs_rest"

class NGram(Enum):
    UNI_GRAM = "unigram"
    BI_GRAM = "bigram"
    TRI_GRAM = "trigram"
    MULTI = "multi"

class SudachiMode(Enum):
    A = "A"
    B = "B"
    C = "C"

class DocClass(BaseModel):
    class_name: str
    docs: list[str]

class CTfidf(BaseModel):
    class_name: str
    token_names: list[str]
    c_tfidf_values: list[float]

class TokenSplitter:
    def __init__(self, n_gram: NGram, sudachi_mode: SudachiMode) -> None:
        self.n_gram: NGram = n_gram
        self.tokenizer_obj = dictionary.Dictionary().create()
        
        match sudachi_mode:
            case SudachiMode.A:
                self.sudachi_mode = tokenizer.Tokenizer.SplitMode.A
            case SudachiMode.B:
                self.sudachi_mode = tokenizer.Tokenizer.SplitMode.B
            case SudachiMode.C:
                self.sudachi_mode = tokenizer.Tokenizer.SplitMode.C

    def split(self, text: str) -> list[str]:
        SPECIAL_UNDER_SCORE = "▁"
        match self.n_gram:
            case NGram.UNI_GRAM:
                tokens = self.tokenizer_obj.tokenize(text, self.sudachi_mode)

            case NGram.BI_GRAM:
                tokens = self.tokenizer_obj.tokenize(text, self.sudachi_mode)
                tokens = [SPECIAL_UNDER_SCORE.join([tokens[i], tokens[i + 1]]) for i in range(len(tokens) - 1)]

            case NGram.TRI_GRAM:
                tokens = self.tokenizer_obj.tokenize(text, self.sudachi_mode)
                tokens = [SPECIAL_UNDER_SCORE.join([tokens[i], tokens[i + 1], tokens[i + 2]]) for i in range(len(tokens) - 2)]

        return tokens


class CTfidfCalculator:
    def __init__(self, n_gram: NGram = NGram.UNI_GRAM, sudachi_mode: SudachiMode = SudachiMode.C, mode: Mode = Mode.ONE_VS_REST) -> None:
        self.n_gram = n_gram
        self.sudachi_mode = sudachi_mode
        self.mode = mode 
        self.doc_classes: list[DocClass] = []
        self.c_tfidfs: list[CTfidf] = []

    def add_doc_class(self, docs: list[str], class_name: str) -> None:
        self.doc_classes.append(DocClass(class_name=class_name, docs=docs))

    def calculate_c_tfidf(self, n_gram: NGram | None = None, sudachi_mode: SudachiMode | None = None,  mode: Mode | None = None) -> list[str]:
        if n_gram:
            self.n_gram = n_gram
        if sudachi_mode:
            self.sudachi_mode = sudachi_mode
        if mode:
            self.mode = mode

        token_splitter = TokenSplitter(n_gram=self.n_gram, sudachi_mode=self.sudachi_mode)
        corpus: list[list[str]] = [token_splitter.split(doc) for doc_class in self.doc_classes for doc in doc_class.docs]

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(corpus)
        c_tfidf_values = tfidf_vectorizer.transform(corpus)
        c_tfidf_tokens = tfidf_vectorizer.vocabulary_

        c_tfidf: list[CTfidf] = []

        for class_index, c_tfidf_values_for_one_class in enumerate(c_tfidf_values):
            class_name = self.doc_classes[class_index].class_name
            token_ctfidf = sorted(list(zip(c_tfidf_tokens, c_tfidf_values_for_one_class)), key=lambda x: x[1], reverse=True)
            c_tfidf.append(CTfidf(class_name=class_name,
                                              token_names=[token_name for token_name, _ in token_ctfidf],
                                              c_tfidf_values=[c_tfidf_value for _, c_tfidf_value in token_ctfidf]))
            
        return c_tfidf
    