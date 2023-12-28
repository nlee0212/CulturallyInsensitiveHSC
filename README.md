# Culturally Insensitive HSC

This repository is for '[Hate Speech Classifiers are Culturally Insensitive](https://aclanthology.org/2023.c3nlp-1.5/)' @ C3NLP 2023.

### For Data Processing
`utils.py`: where all the functions utilized throughout the experiment are  
`main.py`: file to execute when performing translation or comparing the cosine similarity

To translate and calculate the cosine-similarity between the original and the RTT, execute as:
```
$ python main.py
```

### For Hate Speech Classifier Training
`finetune/config.py`: configurations and hyperparameters for the experiment   
`finetune/dataset.py`: python classes used for dataset preprocessing & utilization for the experiment   
`finetune/finetune.py`: finetune models following the configurations from `config.py`   
 
To perform finetuning or transfer learning, within the `finetune/` directory, execute as:
```
$ python finetune.py
```
