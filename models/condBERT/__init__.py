import os
import sys
from condbert import CondBertRewriter
from choosers import EmbeddingSimilarityChooser
from multiword.masked_token_predictor_bert import MaskedTokenPredictorBert
import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import pickle
from tqdm.auto import tqdm, trange

import numpy as np

from flair.data import Sentence
from flair.embeddings import WordEmbeddings
import gensim

from importlib import reload

BERT_WEIGHTS = 'ru_cond_bert_geotrend/checkpoint-9000/pytorch_model.bin'
VOCAB_DIRNAME = 'ru_vocabularies_geotrend' 
    

class condBERT:
    
    def __init__(self, device='cuda', from_pretrained=True):
        def adjust_logits(logits):
            return logits - token_toxicities * 100
        
        model_name = 'Geotrend/bert-base-ru-cased'
        tokenizer_ru = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)

        if from_pretrained:
            print('Loading pre-trained weights.')
            if not os.path.isdir(BERT_WEIGHTS.split('/')[0]):
                os.system('gdown --id 1z5UlXYpZPBC0hlP6W8EMdcgCZmpO5lPg && unzip ru_cond_bert_geotrend.zip')
            
            model_dict = torch.load(BERT_WEIGHTS, map_location=device)
            model.load_state_dict(model_dict, strict=False)
            
        model.to(device);
            
        if not os.path.isdir(VOCAB_DIRNAME):
            print('Loading pre-calculated vocabularies.')
            os.system('gdown --id 1BZTmXqvJe-R0MzYbY6QD7KaySLiM38Em && unzip ru_vocabularies_geotrend.zip')
            
        with open(VOCAB_DIRNAME + "/negative-words.txt", "r") as f:
            s = f.readlines()
        negative_words = list(map(lambda x: x[:-1], s))

        with open(VOCAB_DIRNAME + "/positive-words.txt", "r") as f:
            s = f.readlines()
        positive_words = list(map(lambda x: x[:-1], s))
        
        with open(VOCAB_DIRNAME + '/word2coef.pkl', 'rb') as f:
            word2coef = pickle.load(f)
        
        token_toxicities = []
        with open(VOCAB_DIRNAME + '/token_toxicities.txt', 'r') as f:
            for line in f.readlines():
                token_toxicities.append(float(line))
        token_toxicities = np.array(token_toxicities)
        token_toxicities = np.maximum(0, np.log(1/(1/token_toxicities-1)))   # log odds ratio

        # discourage meaningless tokens
        for tok in ['.', ',', '-']:
            token_toxicities[tokenizer_ru.encode(tok)][1] = 3
            
        predictor = MaskedTokenPredictorBert(model, tokenizer_ru, max_len=250, device=device, label=0, contrast_penalty=0.0, logits_postprocessor=adjust_logits)

        self.editor = CondBertRewriter(
            model=model,
            tokenizer=tokenizer_ru,
            device=device,
            neg_words=negative_words,
            pos_words=positive_words,
            word2coef=word2coef,
            token_toxicities=token_toxicities,
            predictor=predictor,
        )
        
        return
    
    
    def detoxify(self, text):
        
        return self.editor.translate(text, prnt=False)