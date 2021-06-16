# ruDetoxifier

Code and data from the paper "Methods for De-toxification of Texts on Social Media for Russian Language".

***

## Inference Example
In `notebooks` you can find notebook for detoxification models launch.

## Metrics

Script `ru_metric.py` include all three parts of Automatic Evaluation:
- style transfer accuracy (STA);
- content preservation (CS, BLEU, WO);
- grammatically correctness (PPL);

Aggregation of metrics is calculated with GM.

***

## Data

Folder `data` consists of all used train datasets, test data and naive example of style transfer result:
- `data/train`: RuToxic dataset, list of Russian rude words, and 200 samples of parallel sentences that were used for ruGPT fine-tuning;
- `data/test`: 10,000 samples that were used for approaches evaluation;
- `data/results`: example of style transfer output format illustrated with naive duplication.
