import numpy as np

from flair.data import Sentence
from flair.embeddings import WordEmbeddings


def cosine(v1, v2):
    return np.dot(v1, v2) / np.sqrt(sum(v1**2) * sum(v2**2) + 1e-10)


class EmbeddingSimilarityChooser:
    def __init__(self, sim_coef=100, tokenizer=None):
        self.glove_embedding = WordEmbeddings('glove')
        self.sim_coef = sim_coef
        self.tokenizer = tokenizer

    def embed(self, text):
        toks = self.glove_embedding.embed(Sentence(text))[0]
        return np.mean([t.embedding.cpu().numpy() for t in toks], axis=0)
    
    def decode(self, tokens):
        if isinstance(tokens, str):
            return tokens
        if self.tokenizer:
            return self.tokenizer.convert_tokens_to_string(tokens)
        return ' '.join(tokens).replace(' ##', '')

    def __call__(self, hypotheses, original=None, scores=None, **kwargs):
        e = self.embed(self.decode(original))
        candidates = [
            (fill_words, score, cosine(e, self.embed(self.decode(fill_words)))) 
            for fill_words, score in zip(hypotheses, scores)
        ]
        candidates = sorted(candidates, key=lambda x: x[1] + x[2] * self.sim_coef, reverse=True)
        return candidates[0][0]

    
class RuEmbeddingSimilarityChooser:
    def __init__(self, sim_coef=100, tokenizer=None):
        self.glove_embedding = WordEmbeddings('glove')
        self.sim_coef = sim_coef
        self.tokenizer = tokenizer

    def embed(self, text):
        toks = self.glove_embedding.embed(Sentence(text))[0]
        return np.mean([t.embedding.cpu().numpy() for t in toks], axis=0)
    
    def decode(self, tokens):
        if isinstance(tokens, str):
            return tokens
        if self.tokenizer:
            return self.tokenizer.convert_tokens_to_string(tokens)
        return ' '.join(tokens).replace(' ##', '')

    def __call__(self, hypotheses, original=None, scores=None, **kwargs):
        e = self.embed(self.decode(original))
        candidates = [
            (fill_words, score, cosine(e, self.embed(self.decode(fill_words)))) 
            for fill_words, score in zip(hypotheses, scores)
        ]
        candidates = sorted(candidates, key=lambda x: x[1] + x[2] * self.sim_coef, reverse=True)
        return candidates[0][0]