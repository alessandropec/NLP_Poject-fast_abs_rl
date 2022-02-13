""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from cytoolz import identity, concat, curry
from toolz.sandbox.core import unzip

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp
from data.data import CnnDmDataset

from data.batcher import tokenize

from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor
from statistics import mean
from metric import compute_rouge_n, compute_rouge_l_summ

DATA_DIR = 'path/to/processed/data/to/be/evaluated'

def coll(batch):
        art_batch, abs_batch = unzip(batch)
        art_sents = list(filter(bool, map(tokenize(None), art_batch)))
        abs_sents = list(filter(bool, map(tokenize(None), abs_batch))) 
        return art_sents, abs_sents

class EvalDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split, max_sent):
        super().__init__(split, DATA_DIR)
        self._max_sent = max_sent

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        if self._max_sent is not None:
            extracts = js_data['extracted']
            art_sents = js_data['article'][:self._max_sent]
            abs_sents = []
            cleaned_extracts = list(filter(lambda e: e < len(art_sents), extracts))
            for (j,e1), e2 in zip(enumerate(extracts), cleaned_extracts):
                if e1 == e2:
                    abs_sents.append(js_data['gold'][j])
        else:
            art_sents = js_data['article']
            abs_sents = js_data['gold']
        return art_sents, abs_sents

def decode_eval(save_path, model_dir, split, max_sent, batch_size, max_len, cuda, coll_fn=coll):
    start = time()
    # setup model
    with open(join(model_dir+'/meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        abstractor = identity
    else:
        abstractor = Abstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)

    extractor = RLExtractor(model_dir, cuda=cuda)

    # setup loader
    dataset = EvalDataset('val', max_sent)

    n_data = len(dataset)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=coll_fn
    )
    # prepare save paths and logs
    os.makedirs(join(save_path, 'outputs'))
    dec_log = {}
    dec_log['abstractor'] = meta['net_args']['abstractor']
    dec_log['extractor'] = meta['net_args']['extractor']
    dec_log['rl'] = True
    dec_log['split'] = split
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding
    i = 0
    avg_reward = {"rouge-2": [], "rouge-1": [], "rouge-L": []}
    with torch.no_grad():
        for raw_article_batch, val_batch in loader:
            #tokenized_article_batch = map(tokenize(None), raw_article_batch)
            #tokenized_gold_batch = map(tokenize(None), gold_batch)
            ext_arts = []
            ext_inds = []
            for raw_art_sents in raw_article_batch:
                ext = extractor(raw_art_sents)[:-1]  # exclude EOE
                if not ext:
                    # use top-5 if nothing is extracted
                    # in some rare cases rnn-ext does not extract at all
                    ext = list(range(5))[:len(raw_art_sents)]
                else:
                    ext = [i.item() for i in ext]
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += [raw_art_sents[i] for i in ext]
            
            dec_outs = abstractor(ext_arts)
            #assert i == batch_size*i_debug
            for (j, n), gold in zip(ext_inds, val_batch):
                for g,sent in enumerate(gold):
                    sent = sent[1:-1] #remove sos/eos
                    gold[g] = sent
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                #evaluation
                avg_reward['rouge-2'].append(compute_rouge_n(list(concat(dec_outs)),list(concat(gold)), n=2))
                avg_reward['rouge-1'].append(compute_rouge_n(list(concat(dec_outs)),list(concat(gold)), n=1))            
                avg_reward['rouge-L'].append(compute_rouge_l_summ(dec_outs, gold))
                with open(join(save_path, f'outputs/{i}.txt'),
                          'w', encoding="utf-8") as f:
                    f.write('\n'.join(decoded_sents))
                i += 1

                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i/n_data*100,
                    timedelta(seconds=int(time()-start))), end='')
                #avg_reward['rouge-2'] /= (i/100)
                #avg_reward['rouge-L'] /= (i/100)


    print('Avg rouge-2: ',mean(avg_reward['rouge-2']))
    print('Avg rouge-1: ',mean(avg_reward['rouge-1']))
    print('Avg rouge-L: ',mean(avg_reward['rouge-L']))
            
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--model_dir', help='root of the full model')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--split', action='store', help='use split set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=2,
                        help='batch size of faster decoding')
    parser.add_argument('--max_sent', type=int, action='store', default=None,
                        help='max num of sents for decoding')
    parser.add_argument('--max_dec_word', type=int, action='store', default=50,
                        help='maximum words to be decoded for the abstractor')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() 

    #data_split = 'test' if args.test else 'val' 

    cuda = torch.cuda.is_available()

    decode_eval(args.path, args.model_dir, args.split, args.max_sent, 
                args.batch, args.max_dec_word, cuda)

