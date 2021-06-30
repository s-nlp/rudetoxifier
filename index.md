# Methods for Detoxification of Texts for the Russian Language (ruDetoxifier)

This repository contains models and evaluation methodology for the detoxification task of Russian texts. [The original paper](https://arxiv.org/abs/2105.09052) "Methods for Detoxification of Texts for the Russian Language" was presented at [Dialogue-2021](http://www.dialog-21.ru/) conference.

***
## Methodology

In our research we tested several approaches:

### Baselines

### detoxGPT

### condBERT

***

## Automatic Evaluation


***
## Results
|Method   |STA↑   |CS   |WO   |BLEU  |PPL  |GM   |
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
|fine-tuned   |0.61   |0.77   |0.22   |0.21   |36.92   |0.23* ± 0.0010   |
|**condBERT**  |   |   |   |   |   |   |
|DeepPavlov zero-shot   |0.53   |0.80   |0.42   |0.61   |668.58   |0.08 ± 0.0006   |
|DeepPavlov fine-tuned   |0.52   |0.86   |0.51   |0.53   |246.68   |0.12 ± 0.0007   |
|Geotrend zero-shot   |0.62   |0.85   |0.54   |0.64   |237.46   |0.13 ± 0.0009   |
|Geotrend fine-tuned   |0.66   |0.86   |0.54   |0.64   |209.95   |0.14 ± 0.0009   |
***

## Inference Example

In this repository, we release two best models:
- **detoxGPT**: 
- 

***

## Evaluation Example

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
