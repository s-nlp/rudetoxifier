import argparse
import numpy as np

from style_trasnfer_accuracy import classify_preds
from content_similarity import get_cosine_sim_corpus, calc_bleu, get_word_overlap_corpus
from language_quality import get_gpt2_ppl_corpus
from aggregation_metric import get_gm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--inputs", help="path to test sentences", required=True)
    parser.add_argument('-p', "--preds", help="path to predictions of a model", required=True)
    parser.add_argument("--batch_size", default=32, type=int)
    
    args = parser.parse_args()
    
    with open(args.inputs, 'r') as input_file, open(args.preds, 'r') as preds_file:
        original_sentences = input_file.readlines()
        transferred_sentences = preds_file.readlines()

    file_name = args.preds.split('.')[0]
    
    results_file = open(file_name+'_results.txt', 'w')

    accuracy_by_sent = classify_preds(args, transferred_sentences)
    acc = np.mean(accuracy_by_sent)
    results_file.write('Style_transfer_accuracy: ' + str(acc) + '\n')
    
    cs = get_cosine_sim_corpus(original_sentences, transferred_sentences)
    results_file.write('Cosine_similarity: ' + str(cs) + '\n')
    
    wo = get_word_overlap_corpus(original_sentences, transferred_sentences)
    results_file.write('Word_Overlap: ' + str(wo) + '\n')
    
    ppl = get_gpt2_ppl_corpus(transferred_sentences)
    results_file.write('Perplexity: ' + str(ppl) + '\n')
    
    gm = get_gm(acc, cs, ppl)
    results_file.write('Geometric_mean: ' + str(gm) + '\n')
    
    bleu = calc_bleu(original_sentences, transferred_sentences)
    results_file.write('BLEU: ' + str(bleu) + '\n')
    
    results_file.close()