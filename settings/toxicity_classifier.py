import os
from settings import ROOT_DIR

PRETR_BERT_PATH = os.path.join(ROOT_DIR, "data", "ru_conversational_cased_L-12_H-768_A-12")
N_CLASSES = 2
BATCH_SIZE = 128
THRESHOLD = 0.5