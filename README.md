# An exploration of hybrid neural doc summarization task
This repository contains the code for our "Deep Natural Language Processing" exam paper at Politecnico of Turin.

Starting from the report of the 2nd classified at [FNS-2021 challenge](http://wp.lancs.ac.uk/cfie/fns2021/) and inspired by the paper [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://arxiv.org/abs/1805.11080), we explore the task of document summarization with an approach that can be used both for abstractive and extractive summaries, we reproduce the result of the cited report and extend those on an abstractive italian dataset, moreover we train different net with different settings in order to explore the different behaviour of the RL model respect to the different rouges score given as reward functions.

You can
1. Preprocess the data
2. Train the models
3. Evaluate the models

*.yipnb files are avalaible [here](https://drive.google.com/drive/folders/1IFG9wRRYJ_SAkOmde7L0A3_axs7eifMG?usp=sharing) you can find a tutorial for the full process and the script to evaluate our models* 

# Dependencies
- **Python 3** (tested on python 3.6)
- [PyTorch](https://github.com/pytorch/pytorch) 0.4.0
    - with GPU and CUDA enabled installation (though the code is runnable on CPU, it would be way too slow)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [cytoolz](https://github.com/pytoolz/cytoolz)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [pyrouge](https://github.com/bheinzerling/pyrouge) (for evaluation)



# Datasets used

1. FNS 2021 donwload [here](https://drive.google.com/drive/folders/1jWEzOjuC47BkrOpM6USdHGqVDisqZefm?usp=sharing) 
2. Italian set of data extracted from WikiLingua download frome [here](https://drive.google.com/drive/folders/1KF0uJWvf1IhDXaMlqlS0jA0CIWmtC8Fd?usp=sharing)
For the italian dataset we refer to it as wiki10k since is an extraction of 10k document and respective summaries.

We report some statistics of the datasets analyzed (Note we use only 450 document for fns due to limited HW resource)

|                    | wiki10k   | wiki10k   | fns-2021  |fns-2021           |
|--------------------|-----------|-----------|-----------|-----------|
|                    | Documents | Summaries | Documents | Summaries |
| tot word           | 11670364  | 1354573   | 21689271  | 1040095   |
| avg words per sent | 19,016    | 9,46      | 36,23     | 29,59     |
| avg words per doc  | 1167,153  | 135,47    | 48198,38  | 2311,32   |
| tot sents          | 613711    | 143098    | 598496    | 35149     |
| avg sents per doc  | 61,37     | 14,31     | 1329,99   | 78,1      |
| tot_docs           | 10000     | 10000     | 450       | 450       |

# Models

In the same folder of dataset are present different already trained model that can be used to run the evaluation.


# Process data (full pipeline)
To run the whole preprocess pipeline use the script inside preprocess folder prepro_pipeline.py, run the following command.

The script generate a folder containing different data processed: the documents, the summaries, the corpus, the BoW. 

Note: the last argument is optionally and used for abstractive summarization since the summaries could have word not present in the documents.

```
python prepro_pipeline.py <PHASE OF PREPROCESSING (all for full pipeline)> <PATH TO RAW DATA>/annual_reports <PATH TO RAW DATA>/gold_summaries (opt)<PATH of folder cotaining both doc and summaries>
```
If you use the Wiki dataset, to extract and generate the raw data run generate_raw_wiki.py use the following command

```
python generate_raw_wiki_ita.py <PATH OF italian.pickle DATASET> <PATH WHERE SAVE THE RAW DATA> <NUMBER OF ARTICLE TO EXTRACT>
```
## Generate labeled data for pointer net training
The script extract_labels.py generate the label by matching the gold summaries sents with the correspondent report sents, for each report generate a json file with the labels and store those inside the destination folder.

Run the following comand to extract labels, specify the source paths for processed report and summaries and the destination path for labeled data
```
python extract_labels.py <PATH OF PROCESSED DOCUMENTS> <PATH OF PROCESSED GOLD SUMMARIES> <PATH WHERE SAVE THE LABELED SAMPLES>
```

# Train models

## Train gensim Word2Vec model
The script train_w2v.py inside preprocess folder read the whole preprocessed corpus and use it to train a gensim word2vec model then save it.

Run the following comand to train the word2vec model
```
python train_w2v.py <PATH OF CORPUS PROCeSSED> <PATH WHERE TO SAVE THE MODEL (specify the model name)>
```
## Train extractor
Use the script train_extractor_ml.py run the following command
```
python train_extractor_ml.py --data_dir=<PATH TO POINTER DATA SPLITTED> --path=<PATH WHERE TO SAVE THE MODEL> --w2v=<PATH OF W2V MODEL>
```

### Extract data for abstractor training 
To train the abstractor we use the data extracted from the extractor, run the following command to make abstractor data

```
python generate_extractor_outputs.py --dir=<PATH OF LABELLED DATA (train or val)> --w2v_file=<PATH OF W2V FILE> --extractor_model=<PATH OF EXTRACTOR MODEL FOLDER> --output_dir=<PATH WHERE TO SAVE THE EXTRACTED DATA>
```

## Train abstractor
Use the script train_abstractor.py run the following command
```
python train_abstractor.py --data_dir=<PATH TO ABST DATA SPLITTED> --path=<PATH WHERE TO SAVE THE MODEL> --w2v=<PATH OF W2V MODEL>
```

## Train full rl model
Use the script train_full_rl.py run the following command

The arguments --reward=avg_rouges sets the reward function as the average of rouge-1, rouge-2, rouge-L, by default the reward is only the rouge-2

```
python train_full_rl.py --data_dir=<PATH OF ABSTRACTOR DATA> --path<PATH WHERE TO SAVE THE MODEL> --ext_dir=<EXTRACTOR MODEL PATH> --abs_dir=<ABSTRACTOR MODEL PATH> --reward=avg_rouges       
```

# Make evaluation
Use the script decode_eval.py to get the results, the validation data are used.
```
python decode_eval.py --path=<PATH WHERE TO SAVE THE EXTRACTION> --model_dir=<PATH OF RL MODEL> --data_dir=<PATH OF DIR OF LABELLED DATA>      
```
# Our results
We train different models on different datasets and with different settings (columns) and evaluate those with the rouge-1, rouge-2 and rouge-L computed for each generated summary against the the reference and averaged over the number of documents.

For the full settings of the model refer to the appendix in the paper.

|             | wiki rl rouge-2 | wiki rl rouge-L | wiki rl avg-rouges | fns rl rouge-2 | fns rl rouge-L | fns rl avg-rouges |
|-------------|-----------------|-----------------|--------------------|----------------|----------------|-------------------|
| avg rouge-2 | 0,0282          | 0,0273          |                    | 0,0413         | 0,218          | 0,0732            |
| avg rouge-1 | 0,154           | 0,149           |                    | 0,139          | 0,348          | 0,237             |
| avg rouge-L | 0,152           | 0,145           |                    | 0,134          | 0,344          | 0,228             |


