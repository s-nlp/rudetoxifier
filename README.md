# ruDetoxifier

Code and data from the paper "Methods for Detoxification of Texts for the Russian Language" by Daryna Dementieva, Daniil Moskovskiy, Varvara Logacheva, David Dale,
Olga Kozlova, Nikita Semenov, and Alexander Panchenko.

***

## Inference Example
In `notebooks` you can find notebooks for detoxification models launch:
- detoxGPT example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mc4Gn0bhFtACpqqFnzuz2L5NeTR8HXdP?usp=sharing)
- condBERT example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yOL-GO22P5vEPxYQ_Wv-A0hvv3XJUw-r?usp=sharing)

***

## Metrics

Script `ru_metric.py` include all three parts of Automatic Evaluation:
- style transfer accuracy (STA): you can obtain toxicity classifier for the Russian language [here](https://drive.google.com/file/d/1hP820N3FddHJPSxM1BxMxV2N_NDfpjgo/view?usp=sharing);
- content preservation (CS, BLEU, WO);
- grammatically correctness (PPL);

Aggregation of metrics is calculated with GM.

***

## Data

Folder `data` consists of all used train datasets, test data and naive example of style transfer result:
- `data/train`: RuToxic dataset, list of Russian rude words, and 200 samples of parallel sentences that were used for ruGPT fine-tuning;
- `data/test`: 10,000 samples that were used for approaches evaluation;
- `data/results`: example of style transfer output format illustrated with naive duplication.
