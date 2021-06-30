import torch

from isanlp.annotation_repr import CSentence

import numpy as np
import copy
from bisect import bisect
from heapq import nlargest
from tqdm import tqdm

from predict_utils import find_bpe_position_by_offset

import gc


import logging
logger = logging.getLogger('usem-experiments')


def get_masked_tokens_from_tagged_text(tagged_text):
    chunks = tagged_text.split('__')
    
    masks = []
    curr_offset = 0
    clean_text = ''
    for chunk_num, chunk in enumerate(chunks):
        if chunk_num % 2 == 1:
            masks.append((curr_offset, curr_offset + len(chunk)))
            
        curr_offset += len(chunk)
        clean_text += chunk
        
    return masks, clean_text
    

def preprocess_tagged_text(t_text, ppl):
    masked_tokens, clean_text = get_masked_tokens_from_tagged_text(t_text)
    logger.debug(f'Clean text: {clean_text}')

    lng_ann = ppl(clean_text)
    sentences = [CSentence(lng_ann['tokens'], sent) for sent in lng_ann['sentences']]
    if masked_tokens == []:  # just to handle sentences where there is no masked token, always mask the first token
        masked_pos = (0, [0])
        
    else:
        masked_pos = find_bpe_position_by_offset([[(word.begin, word.end) for word in sent] 
                                              for sent in sentences], 
                                             masked_tokens[0])
    
    return masked_pos[1][0], sentences[masked_pos[0]]
    
    
def process_batch(b_text, predictor, ppl, *args, **kwargs):
    l_masked_position = []
    l_bpe_tokens = []
    l_masked_tokens = []
    for j in range(len(b_text)):
        masked_position, tokens = preprocess_tagged_text(b_text[j], ppl)

        l_masked_position.append(masked_position)
        l_bpe_tokens.append(tokens)

        l_masked_tokens.append(tokens[masked_position])
        logger.info(f'Masked token: {tokens[masked_position]}')

    pred_tokens, scores = predictor(l_bpe_tokens, l_masked_position, *args, **kwargs)   
    return pred_tokens, scores, l_masked_tokens
    

def analyze_tagged_text(tagged_text, 
                        predictor, 
                        ppl,
                        batch_size=10,
                        progress_bar=None,
                        n_units=0,
                        n_top=5,
                        fix_multiunit=True,
                        mask_token=True,
                        n_tokens=[1],
                        max_multiunit=10,
                        multiunit_lookup=100,
                        contexts=None):
    """
    - tagged_text (str): a text with a masked tokens highlighted as "__something__" .
    - predictor (object): a predictor object, look in masked_token_predictor_bert.py
    - batch_size (int): a number of examples in an inference batch of the predictor model (it affects speed 
    of generation of single words and multi word phrases)
    - progress_bar(tqdm): use standard tqdm or tqdm_notebook. To eliminate the progress bar, keep it None.
    - n_units(int): number of units to be predicted at maximum during a word generation.
    - n_top(int): maximum number of predictions at all.
    - fix_multiunit(bool): if True the multiunit tokens will look as a regular token without ##. If False you can see
    where the multiunit tokens were predicted.
    - mask_token(bool): tells wheather predictor should mask a token or keep it during the inference (only for single unit tokens).
    - n_tokens(List): a list with phrases lengths. Examples: If you want to predict only single words, use [1]. If you wnat additionally two word predictions,
    use [1, 2]. If you want only 2 word predictions use [2], etc.
    """
    
    if type(tagged_text) is not list:
        tagged_text = [tagged_text]
        
    final_pred_tokens = []
    final_scores = []
    final_masked_tokens = []
    progress_bar = (lambda a: a) if progress_bar is None else progress_bar
    
    for i in progress_bar(range(0, len(tagged_text), batch_size)):
        b_text = tagged_text[i : i + batch_size]
        # for melamud
        if contexts:
            b_contexts = contexts[i: i + batch_size]
        else:    
            b_contexts = None
            
        b_pred_tokens, b_scores, b_masked_tokens = process_batch(b_text, predictor, ppl, 
                                                                 n_units=n_units, 
                                                                 n_top=n_top, 
                                                                 fix_multiunit=fix_multiunit,
                                                                 mask_token=mask_token,
                                                                 n_tokens=n_tokens,
                                                                 multiunit_lookup=multiunit_lookup,
                                                                 max_multiunit=max_multiunit,
                                                                 Cs=b_contexts)
        final_pred_tokens += b_pred_tokens
        final_scores += b_scores
        final_masked_tokens += b_masked_tokens
        
        if (i % 5) == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if len(final_pred_tokens) == 1:
        final_pred_tokens = final_pred_tokens[0]
        final_scores = final_scores[0]
        final_masked_tokens = final_masked_tokens[0]
    return final_pred_tokens, final_scores, final_masked_tokens
