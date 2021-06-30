from tqdm import tqdm
import os
from settings.toxicity_classifier import *
from deeppavlov.dataset_readers.basic_classification_reader import BasicClassificationDatasetReader
from deeppavlov.dataset_iterators.basic_classification_iterator import BasicClassificationDatasetIterator
from deeppavlov.models.preprocessors.bert_preprocessor import BertPreprocessor
from deeppavlov.models.bert.bert_classifier import BertClassifierModel
from deeppavlov.models.classifiers.proba2labels import Proba2Labels
from sklearn.metrics import accuracy_score

prob2labels = Proba2Labels(max_proba=True)

def get_style_transfer_acc(results_csv_path):
    bert_preprocessor = BertPreprocessor(vocab_file=os.path.join(PRETR_BERT_PATH, "vocab.txt"),
                                         do_lower_case=False,
                                         max_seq_length=256)
    bert_classifier = BertClassifierModel(
        n_classes=N_CLASSES,
        return_probas=True,
        one_hot_labels=True,
        bert_config_file=os.path.join(PRETR_BERT_PATH, "bert_config.json"),
        pretrained_bert=os.path.join(PRETR_BERT_PATH, "bert_model.ckpt"),
        save_path="toxic_classifier/model",
        load_path="toxic_classifier/model",
        keep_prob=0.5,
        learning_rate=1e-05,
        learning_rate_drop_patience=5,
        learning_rate_drop_div=2.0
    )

    reader = BasicClassificationDatasetReader()
    data = reader.read(data_path="./",
                       train=results_csv_path, valid=results_csv_path,
                       x="comment", y="toxic")

    iterator = BasicClassificationDatasetIterator(data)

    y_valid_preds = []
    y_valid_texts = []
    progress_bar_valid = tqdm(total=int(10000 / BATCH_SIZE), desc='valid')

    for instance in list(tqdm._instances):
        tqdm._decr_instances(instance)

    for x, y in iterator.gen_batches(batch_size=BATCH_SIZE,
                                     data_type="valid"):
        y_valid_texts.extend(x)

        y_valid_pred = bert_classifier(bert_preprocessor(x))
        y_valid_preds.extend(y_valid_pred)

        progress_bar_valid.update()

    y_valid_preds_adjusted = [list(i)[1] for i in y_valid_preds]
    y_valid_preds_adjusted = [1 if y_pred >= THRESHOLD else 0 for y_pred in y_valid_preds_adjusted]

    y_true = [0] * len(y_valid_preds_adjusted)

    return accuracy_score(y_true, y_valid_preds_adjusted)