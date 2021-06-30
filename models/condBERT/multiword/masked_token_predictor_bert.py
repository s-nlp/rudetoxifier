import torch

import numpy as np
import copy
import bisect
import heapq
import scipy

from predict_utils import find_bpe_position_by_offset
from torch.utils.data import DataLoader

from tensorflow.keras.preprocessing.sequence import pad_sequences

import logging
logger = logging.getLogger('usem-experiments')

            
def bpe_tokenize(bpe_tokenizer, sentence):
    sent_bpe_tokens = []
    sent_bpe_offsets = []
    for token in sentence:
        token_bpes = bpe_tokenizer.tokenize(token.text)
        sent_bpe_offsets += [(token.begin, token.end) for _ in range(len(token_bpes))]
        sent_bpe_tokens += token_bpes
    
    return sent_bpe_tokens, sent_bpe_offsets


def nlargest_indexes(arr, n_top):
    arr_ids = np.argpartition(arr, -n_top)[-n_top:]
    sel_arr = arr[arr_ids]
    top_ids = arr_ids[np.argsort(-sel_arr)]
    return top_ids


def remove_masked_token_subwords(masked_position, bpe_tokens, bpe_offsets):
    """
    If the masked token has been tokenied into multiple subwords: like dieting-->diet and ##ing
    keep the first subword and remove others.
    """
    
    logger.debug(f'bpe tokens: {bpe_tokens}')
    logger.debug(f'bpe offsets: {bpe_offsets}')
    
    if len(masked_position[1]) > 1:
        indexes_to_del = masked_position[1][1:]
        del bpe_tokens[masked_position[0]][indexes_to_del[0] : indexes_to_del[-1] + 1]
        del bpe_offsets[masked_position[0]][indexes_to_del[0] : indexes_to_del[-1] + 1]

    
    masked_position = (masked_position[0], masked_position[1][0]) # TODO: leave masked_token as a list
    
    logger.debug(f'bpe offsets: {str(bpe_tokens)}')
    logger.debug(f'bpe offsets: {str(bpe_offsets)}')

    return masked_position,  bpe_tokens, bpe_offsets


def merge_sorted_results(objects_left, scores_left, 
                         objects_right, scores_right, max_elems):   
    result_objects = []
    result_scores = []
    
    j = 0
    i = 0
    while True:
        if (len(result_scores) == max_elems):
            break
            
        if i == len(scores_left):
            result_objects += objects_right[j : j + max_elems - len(result_scores)]
            result_scores += scores_right[j : j + max_elems - len(result_scores)]
            break
        
        if j == len(scores_right):
            result_objects += objects_left[i : i + max_elems - len(result_scores)]
            result_scores += scores_left[i : i + max_elems - len(result_scores)]
            break
        
        if scores_left[i] > scores_right[j]:
            result_objects.append(objects_left[i])
            result_scores.append(scores_left[i])
            i += 1
        else:
            result_objects.append(objects_right[j])
            result_scores.append(scores_right[j])
            j += 1
    
    return result_objects, result_scores


class MaskedTokenPredictorBert:
    def __init__(
        self, model, bpe_tokenizer, max_len=250, mask_in_multiunit=False, device=None, label=0,
        logits_postprocessor=None, contrast_penalty=0,
    ):
        self._model = model
        self._bpe_tokenizer = bpe_tokenizer
        self._max_len = max_len
        self._mask_in_multiunit = mask_in_multiunit
        self.device = device or torch.device('cuda')
        self.label = label
        self.logits_postprocessor = logits_postprocessor
        self.contrast_penalty = contrast_penalty
    
    def __call__(self, sentences, masked_position, **kwargs):
        if type(masked_position) is not list:
            bpe_tokens = [bpe_tokens]
            masked_position = [masked_position]
        
        b_masked_pos = []
        b_bpe_tokens = []
        for sent, mask_pos in zip(sentences, masked_position):
            bpe_tokens, bpe_offsets = bpe_tokenize(self._bpe_tokenizer, sent)
        
            masked_position = find_bpe_position_by_offset([bpe_offsets], 
                                                          (sent[mask_pos].begin, 
                                                           sent[mask_pos].end)) 

            masked_position, bpe_tokens, _ = remove_masked_token_subwords(masked_position, 
                                                                          [bpe_tokens], 
                                                                          [bpe_offsets])
            bpe_tokens = bpe_tokens[0]

            logger.debug(f'Bpe tokens: {bpe_tokens}')
            
            b_bpe_tokens.append(bpe_tokens)
            b_masked_pos.append(masked_position[1])
        
        return self.generate(b_bpe_tokens, b_masked_pos, **kwargs)
        
    def generate(self, b_bpe_tokens, b_masked_pos,
                 mask_token=True, 
                 n_top=5, 
                 n_units=1, 
                 n_tokens=[1],
                 fix_multiunit=True,
                 beam_size=10,
                 multiunit_lookup=100,
                 max_multiunit=10,
                 **kwargs
    ):
        result_preds = [[] for _ in range(len(b_bpe_tokens))]
        result_scores = [[] for _ in range(len(b_bpe_tokens))]
        if 1 in n_tokens:
            result_preds, result_scores = self.predict_single_word(b_bpe_tokens, b_masked_pos, 
                                            mask_token=mask_token, 
                                            n_top=n_top, 
                                            n_units=n_units, 
                                            multiunit_lookup=multiunit_lookup,
                                            fix_multiunit=fix_multiunit,
                                            max_multiunit=max_multiunit)
        
        for n_t in n_tokens:
            if n_t == 1:
                continue
            
            pred_tokens, pred_scores = self.predict_token_sequence(b_bpe_tokens, b_masked_pos, 
                                               mask_token=mask_token, 
                                               n_top=n_top, 
                                               n_units=n_units, 
                                               seq_len=n_t,
                                               multiunit_lookup=multiunit_lookup,
                                               fix_multiunit=fix_multiunit,
                                               beam_size=beam_size,
                                                max_multiunit=max_multiunit)
            
            for i in range(len(b_bpe_tokens)):
                result_preds[i], result_scores[i] = merge_sorted_results(result_preds[i], result_scores[i],
                                                                         pred_tokens[i], pred_scores[i],
                                                                         n_top)
        
        return result_preds, result_scores
    
    def predict_single_unit(self, bpe_tokens, masked_position, 
                            mask_token, n_top):
        bpe_tokens = copy.deepcopy(bpe_tokens)

        max_len = min([max(len(e) for e in bpe_tokens) + 2, self._max_len])
        token_ids = []
        for i in range(len(bpe_tokens)):
            bpe_tokens[i] = bpe_tokens[i][:max_len - 2]

            if mask_token:
                bpe_tokens[i][masked_position[i]] = '[MASK]'

            bpe_tokens[i] = ['[CLS]'] + bpe_tokens[i] + ['[SEP]']
            logger.debug(f'Masked BPE tokens: {bpe_tokens[i]}')

            token_ids.append(self._bpe_tokenizer.convert_tokens_to_ids(bpe_tokens[i]))

        token_ids = pad_sequences(token_ids, maxlen=max_len, dtype='long', 
                                  truncating='post', padding='post')
        attention_masks_tensor = torch.tensor(token_ids > 0).long().to(self.device)
        tokens_tensor = torch.tensor(token_ids).to(self.device)
        
        segments_ids = np.ones_like(token_ids, dtype=int) * self.label
        segments_tensor = torch.tensor(segments_ids).to(self.device)
        
        self._model.eval()
        with torch.no_grad():
            target_sent = self._model(tokens_tensor, segments_tensor, attention_masks_tensor)[0]

        # target_sent = torch.log_softmax(target_sent, -1)

        if self.contrast_penalty:
            # todo: make it work with multiple words (it breaks hypotheses somehow)
            with torch.no_grad():
                another = self._model(tokens_tensor, 1 - segments_tensor, attention_masks_tensor)[0]
            diff =  torch.softmax(target_sent, -1) - self.contrast_penalty * torch.softmax(another, -1)
            target_sent = torch.log(torch.clamp(diff, 1e-20))

        target_sent = target_sent.detach().cpu().numpy()

        final_top_scores = []
        final_top_tokens = []
        for i in range(target_sent.shape[0]):
            row = target_sent[i]
            idx = masked_position[i]
            logits = row[idx + 1]
            logits = self.adjust_logits(logits)
            top_ids = nlargest_indexes(logits, n_top)
            top_scores = [target_sent[i][masked_position[i] + 1][j] for j in top_ids]
            top_tokens = self._bpe_tokenizer.convert_ids_to_tokens(top_ids)
            
            final_top_scores.append(top_scores)
            final_top_tokens.append(top_tokens)

        return final_top_tokens, final_top_scores
    
    def adjust_logits(self, logits):
        if self.logits_postprocessor:
            return self.logits_postprocessor(logits)
        return logits
    
    def predict_single_word(self, bpe_tokens, masked_position, 
                            mask_token, 
                            n_top, 
                            n_units, 
                            fix_multiunit,
                            multiunit_lookup,
                            max_multiunit):
        pred_tokens, scores = self.predict_single_unit(bpe_tokens, 
                                                       masked_position, 
                                                       mask_token=mask_token, 
                                                       n_top=n_top)
        
        final_pred_tokens = []
        final_scores = []
        for j in range(len(pred_tokens)):
            if n_units > 1:
                pred_tokens[j] = list(reversed(pred_tokens[j][:multiunit_lookup]))
                scores[j] = list(reversed(scores[j][:multiunit_lookup]))

                seq_list = self.generate_multiunit_token(masked_position[j], bpe_tokens[j],
                                                         n_top=multiunit_lookup,
                                                         n_units=n_units)

                #for seq in seq_list[ :max_multiunit - 1]:
                for seq in seq_list[:max_multiunit]:
                    seq_pred, seq_scores = seq
                    multiunit_token = '_'.join(seq_pred)
                    if fix_multiunit:
                        multiunit_token = multiunit_token.replace('#', '')
                        multiunit_token = multiunit_token.replace('_', '')

                    #multiunit_score = np.average(seq_scores)
                    multiunit_score = scipy.stats.hmean(seq_scores)

                    ind = bisect.bisect(scores[j], multiunit_score)

                    pred_tokens[j].insert(ind, multiunit_token)
                    scores[j].insert(ind, multiunit_score)

                pred_tokens[j] = list(reversed(pred_tokens[j]))
                scores[j] = list(reversed(scores[j]))

            logger.debug(f'Predicted words: {pred_tokens[j]}')

            #final_pred_tokens.append([e for e in pred_tokens[j] if not e.startswith('##')])
            final_pred_tokens.append(pred_tokens[j][:n_top])
            final_scores.append(scores[j][:n_top])
        
        return final_pred_tokens, final_scores
    
    def generate_multiunit_token(self, masked_position, bpe_tokens,  
                                 n_top, 
                                 n_units):
        final_result = []
        final_result_scores = []

        bpe_tokens = copy.deepcopy(bpe_tokens)
        bpe_tokens.insert(masked_position, '[MASK]')
        predictions, scores = self.predict_single_unit([bpe_tokens], 
                                                       [masked_position + 1], 
                                                       n_top=n_top, 
                                                       mask_token=self._mask_in_multiunit)
        # This will result in "Mama washed the [MASK] __frame__ ."

        predictions = predictions[0]
        scores = scores[0]
        #n_suffix = 0

        good_preds = []

        # TODO: increase speed
        b_bpe_tokens = []
        for i, pred in (e for e in enumerate(predictions) if e[1][0] == '#'):
#             if n_suffix > n_units - 1:
#                 break

            tmp = copy.deepcopy(bpe_tokens)
            tmp[masked_position + 1] = pred
            b_bpe_tokens.append(tmp)
            good_preds.append((i,pred))
            #n_suffix += 1

        if not good_preds:
            return []

        
        loader = DataLoader(b_bpe_tokens, batch_size=10, collate_fn=lambda _: _)
        preds = []
        pred_scores = []
        for batch in loader:
            bb_preds, bb_pred_scores  = self.predict_single_unit(batch, 
                                              [masked_position for _ in range(len(batch))],
                                              mask_token=False, # We do not need masking here, since inserted token will be a mask
                                              n_top=n_top)
            
            preds += bb_preds
            pred_scores += bb_pred_scores

        for i in range(len(preds)):
            result = [preds[i][0], good_preds[i][1]]
            result_scores = [pred_scores[i][0], scores[good_preds[i][0]]] 

            tail, tail_scores = self.generate_from_tail(preds[i][0], b_bpe_tokens[i], masked_position, 
                                                        max_subunits=n_units-2,
                                                        n_top=n_top)
            result = tail + result
            result_scores = tail_scores + result_scores

            final_result.append(result)
            final_result_scores.append(result_scores)

        return list(zip(final_result, final_result_scores))
    
    def generate_from_tail(self, pred, bpe_tokens, masked_position, 
                           max_subunits,
                           n_top):
        result = []
        result_scores = []

        it = 0
        while (pred[0] == '#') and (it < max_subunits):
            bpe_tokens[masked_position] = pred
            bpe_tokens.insert(masked_position, '[MASK]')
            preds, pred_scores = self.predict_single_unit([bpe_tokens], [masked_position], 
                                                          n_top=n_top, 
                                                          mask_token=False)
            pred = preds[0][0]
            result.append(pred)
            result_scores.append(pred_scores[0][0])
            it += 1

        return (list(reversed(result)), 
                list(reversed(result_scores)))
    
    def generate_variants(self, bpe_tokens, mask_pos, 
                          gen_tokens, gen_scores, seq_len):
        batch_size = len(bpe_tokens)
        
        if not gen_tokens:
            yield bpe_tokens, [0.]*batch_size, [[] for _ in range(batch_size)], mask_pos
            return
        
        for var_num in range(len(gen_tokens[0])):
            if not gen_tokens[0][var_num]:
                continue
                
            variant = []
            new_mask = []
            var_t  = []
            var_s = []
            for i in range(batch_size):    
                new_bpe = copy.deepcopy(bpe_tokens[i])
                
                for seq_num in range(len(gen_tokens[i][var_num])):
                    new_bpe[mask_pos[i] + seq_num] = gen_tokens[i][var_num][seq_num]
                
                var_t.append(gen_tokens[i][var_num])
                var_s.append(gen_scores[i][var_num])
                
                new_mask.append(mask_pos[i] + len(gen_tokens[i][var_num]))

                variant.append(new_bpe)
        
            yield variant, var_s, var_t, new_mask
                    
    def update_beam(self, 
                    prev_tokens, prev_score, 
                    new_scores, new_tokens, 
                    gen_scores, gen_tokens):
        for i in range(len(gen_scores)):
            final_gen_score = prev_score + gen_scores[i]
            insert_pos = bisect.bisect(new_scores, final_gen_score)
            
            new_scores.insert(insert_pos, final_gen_score)
            del new_scores[0]
            
            new_tokens.insert(insert_pos, prev_tokens + [gen_tokens[i]])
            if len(new_tokens) > len(new_scores):
                del new_tokens[0]
        
    def predict_token_sequence(self, bpe_tokens, masked_pos, 
                               mask_token, 
                               n_top, 
                               seq_len, 
                               beam_size, 
                               n_units,
                               fix_multiunit,
                               multiunit_lookup,
                               max_multiunit):
        bpe_tokens = copy.deepcopy(bpe_tokens)
        
        batch_size = len(bpe_tokens)
        for i in range(batch_size):
            for seq_num in range(seq_len - 1):
                bpe_tokens[i].insert(masked_pos[i] + 1, '[MASK]')
        
        gen_scores = []
        gen_tokens = []
        for seq_num in range(seq_len):
            gen_scores_seq = [[0. for __ in range(beam_size)] for _ in range(batch_size)]
            gen_tokens_seq = [[[] for __ in range(beam_size)] for _ in range(batch_size)]
            for variant, variant_score, prev_tokens, new_mask in self.generate_variants(bpe_tokens, masked_pos, 
                                                                                        gen_tokens, gen_scores, 
                                                                                        seq_len=seq_len):
                top_tokens, top_scores = self.predict_single_word(variant, new_mask, 
                                                                  mask_token=True, 
                                                                  n_top=n_top, 
                                                                  n_units=n_units,
                                                                  fix_multiunit=fix_multiunit,
                                                                  multiunit_lookup=multiunit_lookup,
                                                                  max_multiunit=max_multiunit)
                
                for i in range(batch_size):
                    self.update_beam(prev_tokens[i], variant_score[i], 
                                     gen_scores_seq[i], gen_tokens_seq[i], 
                                     top_scores[i], top_tokens[i])
            
            gen_tokens = gen_tokens_seq
            gen_scores = gen_scores_seq
                    
        gen_scores = [[(e/seq_len) for e in l] for l in gen_scores]
                
        return ([list(reversed(e)) for e in gen_tokens], 
                 [list(reversed(e)) for e in gen_scores])
