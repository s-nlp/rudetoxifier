# Methods for Detoxification of Texts for the Russian Language (ruDetoxifier)

This repository contains models and evaluation methodology for the detoxification task of Russian texts. [The original paper](https://arxiv.org/abs/2105.09052) "Methods for Detoxification of Texts for the Russian Language" was presented at [Dialogue-2021](http://www.dialog-21.ru/) conference.

📰 **Updates**

Check out **TextDetox** 🤗 https://huggingface.co/collections/textdetox/ -- continuation of ParaDetox project!

**[2025] !!!NOW OPEN!!! TextDetox CLEF2025 shared task: for even more -- 15 languages!** [website](https://pan.webis.de/clef25/pan25-web/text-detoxification.html) 🤗[Starter Kit](https://huggingface.co/collections/textdetox/)

**[2025] COLNG2025**: Daryna Dementieva, Nikolay Babakov, Amit Ronen, Abinew Ali Ayele, Naquee Rizwan, Florian Schneider, Xintong Wang, Seid Muhie Yimam, Daniil Alekhseevich Moskovskiy, Elisei Stakovskii, Eran Kaufman, Ashraf Elnagar, Animesh Mukherjee, and Alexander Panchenko. 2025. ***Multilingual and Explainable Text Detoxification with Parallel Corpora***. In Proceedings of the 31st International Conference on Computational Linguistics, pages 7998–8025, Abu Dhabi, UAE. Association for Computational Linguistics. [pdf](https://aclanthology.org/2025.coling-main.535/)

**[2024]** We have also created versions of ParaDetox in more languages. You can checkout a [RuParaDetox](https://huggingface.co/datasets/s-nlp/ru_paradetox) dataset as well as a [Multilingual TextDetox](https://huggingface.co/textdetox) project that includes 9 languages.

Corresponding papers:
* [MultiParaDetox: Extending Text Detoxification with Parallel Data to New Languages](https://aclanthology.org/2024.naacl-short.12/) (NAACL 2024)
* [Overview of the multilingual text detoxification task at pan 2024](https://ceur-ws.org/Vol-3740/paper-223.pdf) (CLEF Shared Task 2024)

## Inference Example

In this repository, we release two best models **detoxGPT** and **condBERT** (see [Methodology](https://github.com/skoltech-nlp/rudetoxifier#methodology) for more details). You can try detoxification inference example in this [notebook](https://github.com/skoltech-nlp/rudetoxifier/blob/main/notebooks/rudetoxifier_inference.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lSXh8PHGeKTLtuhxYCwHL74qG-V-pkLK?usp=sharing).

## Interactive Demo
[Old Versions] Also, you can test our models via [web-demo](https://detoxifier-nlp-zh.skoltech.ru/) or you can pour out your anger on our [Telegram bot](https://t.me/rudetoxifierbot).

***
## Methodology

In our research, we tested several approaches:

### Baselines
- Duplicate: simple duplication of the input;
- Delete: removal of rude and toxic from pre-defined [vocab](https://github.com/skoltech-nlp/rudetoxifier/blob/main/data/train/MAT_FINAL_with_unigram_inflections.txt);
- Retrieve: retrieval based on cosine similarity between word embeddings from non-toxic part of [RuToxic](https://github.com/skoltech-nlp/rudetoxifier/blob/main/data/train/ru_toxic_dataset.csv) dataset;

### detoxGPT
Based on [ruGPT](https://github.com/sberbank-ai/ru-gpts) models. This method requires [parallel dataset](https://github.com/skoltech-nlp/rudetoxifier/blob/main/data/train/dataset_200.xls) for training. We tested ***ruGPT-small***, ***ruGPT-medium***, and ***ruGPT-large*** models in several setups:
- ***zero-shot***: the model is taken as is (with no fine-tuning). The input is a toxic sentence which we would like to detoxify prepended with the prefix “Перефразируй” (rus. Paraphrase) and followed with the suffix “>>>” to indicate the paraphrasing task
- ***few-shot***: the model is taken as is. Unlike the previous scenario, we give a prefix consisting of a parallel dataset of toxic and neutral sentences.
- ***fine-tuned***: the model is fine-tuned for the paraphrasing task on a parallel dataset.

### condBERT
Based on [BERT](https://arxiv.org/abs/1810.04805) model. This method *does not* require parallel dataset for training. One of the tasks on which original BERT was pretrained -- predicting the word that should was replaced with a \[MASK\] token -- suits delete-retrieve-generate style transfer method. We tested [RuBERT](https://huggingface.co/DeepPavlov/rubert-base-cased-conversational) and [Geotrend](https://huggingface.co/Geotrend/bert-base-ru-cased) pre-trained models in several setups:
- ***zero-shot*** where BERT is taken as is (with no extra fine-tuning);
- ***fine-tuned*** where BERT is fine-tuned on a dataset of toxic and safe sentences to acquire a style-
dependent distribution, as described above.

***

## Automatic Evaluation
The evaluation consists of three types of metrics:
- **style transfer accuracy (STA)**: accuracy based on toxic/non-toxic classifier (we suppose that the resulted text should be in non-toxic style)
- **content preservation**:
  - word overlap (WO);
  - BLEU: accuracy based on n-grams (1-4);
  - cosine similarity (CS): between vectors of texts’ embeddings.
- **language quality**: perplexity (PPL) based on language model.

Finally, **aggregation metric**: geometric mean between STA, CS and PPL.

### Launching

You can run [`ru_metric.py`](https://github.com/skoltech-nlp/rudetoxifier/blob/main/metrics/ru_metric.py) script for evaluation. The fine-tuned weights for toxicity classifier can be found [here](https://drive.google.com/file/d/1WqNOyFegzUWoY7tCMtmtMt9Ct0lhRwHy/view?usp=sharing).

***

## Results

|Method   |STA↑   |CS↑   |WO↑   |BLEU↑  |PPL↓  |GM↑   |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
|**Baselines**
|Duplicate   |0.00   |1.00   |1.00   |1.00   |146.00   |0.05 ± 0.0012   |
|Delete   |0.27   |0.96   |0.85   |0.81   |263.55   |0.10 ± 0.0007   |
|Retrieve   |0.91   |0.85   |0.07   |0.09   |65.74   |0.22 ± 0.0010   |
|**detoxGPT-small**
|zero-shot   |0.93   |0.20   |0.00   |0.00   |159.11   |0.10 ± 0.0005   |
|few-shot   |0.17   |0.70   |0.05   |0.06   |83.38   |0.11 ± 0.0009   |
|fine-tuned   |0.51   |0.70   |0.05   |0.05   |39.48   |0.20 ± 0.0011   |
|**detoxGPT-medium**   |   |   |   |   |   |   |
|fine-tuned   |0.49   |0.77   |0.18   |0.21   |86.75   |0.16 ± 0.0009   |
|**detoxGPT-large**   |   |   |   |   |   |   |
|fine-tuned   |0.61   |0.77   |0.22   |0.21   |**36.92**  |**0.23 ± 0.0010**  |
|**condBERT**  |   |   |   |   |   |   |
|DeepPavlov zero-shot   |0.53   |0.80   |0.42   |0.61   |668.58   |0.08 ± 0.0006   |
|DeepPavlov fine-tuned   |0.52   |0.86   |0.51   |0.53   |246.68   |0.12 ± 0.0007   |
|Geotrend zero-shot   |0.62   |0.85   |0.54   |**0.64**   |237.46   |0.13 ± 0.0009   |
|Geotrend fine-tuned   |**0.66**  |**0.86**  |**0.54**   |0.64   |209.95   |0.14 ± 0.0009   |

***

## Data

Folder `data` consists of all used train datasets, test data and naive example of style transfer result:
- `data/train`: RuToxic dataset, list of Russian rude words, and 200 samples of parallel sentences that were used for ruGPT fine-tuning;
- `data/test`: 10,000 samples that were used for approaches evaluation;
- `data/results`: example of style transfer output format illustrated with naive duplication.

***

## Citation

If you find this repository helpful, feel free to cite our publication:

```
@article{DBLP:journals/mti/DementievaMLDKS21,
  author       = {Daryna Dementieva and
                  Daniil Moskovskiy and
                  Varvara Logacheva and
                  David Dale and
                  Olga Kozlova and
                  Nikita Semenov and
                  Alexander Panchenko},
  title        = {Methods for Detoxification of Texts for the Russian Language},
  journal      = {Multimodal Technol. Interact.},
  volume       = {5},
  number       = {9},
  pages        = {54},
  year         = {2021},
  url          = {https://doi.org/10.3390/mti5090054},
  doi          = {10.3390/MTI5090054},
  timestamp    = {Wed, 15 Dec 2021 10:31:28 +0100},
  biburl       = {https://dblp.org/rec/journals/mti/DementievaMLDKS21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

***

## Contacts

For any questions please contact Daryna Dementieva via [email](mailto:dardem96@gmail.com).
