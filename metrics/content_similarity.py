import numpy as np
from tqdm import tqdm
import gensim
from ufal.udpipe import Model, Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

model = gensim.models.KeyedVectors.load('ru_fasttext/model.model')
modelfile = 'udpipe_syntagrus.model'
model_udpipe = Model.load(modelfile)
process_pipeline = Pipeline(model_udpipe, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

def tokenize(text, tags=False, lemmas=False):
    processed = process_pipeline.process(text)
    content = [l for l in processed.split('\n') if not l.startswith('#')]
    tagged = [w.split('\t') for w in content if w]

    tokens = []
    for token in tagged:
        if token[3] == 'PUNCT':
            continue

        token_res = ''
        if lemmas:
            token_res = token[2]
        else:
            token_res = token[1]
        if tags:
            token_res += '_' + token[3]
        tokens.append(token_res)

    return tokens


def get_sentence_vector(text):
    tokens = tokenize(text, lemmas=True)
    embd = [model[token] for token in tokens]

    return np.mean(embd, axis=0).reshape(1, -1)


def get_cosine_sim(text1, text2):
    try:
        return cosine_similarity(get_sentence_vector(text1), get_sentence_vector(text2))
    except:
        return 0


def get_cosine_sim_corpus(original_sentences, transferred_sentences):
    results = []
    for index in tqdm(range(len(original_sentences))):
        results.append(get_cosine_sim(original_sentences[index], transferred_sentences[index]))

    return np.mean(results)


def get_word_overlap(text1, text2):
    tokens1 = tokenize(text1, lemmas=True)
    tokens2 = tokenize(text2, lemmas=True)

    union = set(tokens1 + tokens2)
    intersection = list(set(tokens1) & set(tokens2))

    return len(intersection) / len(union)


def get_word_overlap_corpus(original_sentences, transferred_sentences):
    results = []
    for index in tqdm(range(len(original_sentences))):
        results.append(get_word_overlap(original_sentences[index], transferred_sentences[index]))

    return np.mean(results)


def get_bleu_corpus(original_sentences, transferred_sentences):
    references = []
    hypothesises = []

    for sentence in original_sentences:
        references.append([[sentence]])
    for sentence in transferred_sentences:
        hypothesises.append([sentence])

    return corpus_bleu(references, hypothesises, weights=[1])


def calc_bleu(inputs, preds):
    bleu_sim = 0
    counter = 0
    print('Calculating BLEU similarity')
    for i in range(len(inputs)):
        if len(inputs[i]) > 3 and len(preds[i]) > 3:
            bleu_sim += sentence_bleu([inputs[i]], preds[i])
            counter += 1

    return float(bleu_sim / counter)