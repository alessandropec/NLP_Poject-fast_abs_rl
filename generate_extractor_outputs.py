import gensim
from toolz.sandbox import unzip
from model.extract import PtrExtractSumm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.optim as optim
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from os.path import join
from decoding import load_best_ckpt
import pickle as pkl

PAD = 0
UNK = 1
START = 2
END = 3

def prepro_fn_extract(max_src_len, max_src_num, batch):
    def prepro_one(sample):
        source_sents, extracts = sample
        tokenized_sents = tokenize(max_src_len, source_sents)[:max_src_num] #Tokenizzo l'intero documento e taglia le frasi lunghe piu di max_src_len
        cleaned_extracts = list(filter(lambda e: e < len(tokenized_sents), #Essendo che il numero di frasi è troncato a max_src_num, rimuovo gli indici che puntano a frasi che sono state tagliate
                                       extracts))
        return tokenized_sents, cleaned_extracts
    batch = list(map(prepro_one, batch))
    batch = [element for element in batch if element[0] and element[1]]
    return batch

def tokenize(max_len, texts):
    return [t.lower().split()[:max_len] for t in texts]

# BATCHIFY

def batchify_fn_extract_ptr_dec(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))
    src_nums = list(map(len, source_lists))
    sources = [pad_batch_tensorize(source,cuda=cuda) for source in source_lists]
    
    fw_args = (sources, src_nums)
    return fw_args


def pad_batch_tensorize(inputs,cuda=False):
    """pad_batch_tensorize
    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    pad = 0
 
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    max_len = max(len(ids) for ids in inputs)
    tensor_shape = (batch_size, max_len)
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        tensor[i, :len(ids)] = tensor_type(ids)
    return tensor

  
def convert_batch_extract_ptr(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        id_sents = conver2id(unk, word2id, source_sents)
        #print(id_sents)
        return id_sents, extracts
    batch = list(map(convert_one, batch))
    return batch

def conver2id(unk, word2id, words_list):
    word2id = defaultdict(lambda: unk, word2id)
    #print(word2id['<SOS>'])
    #print(words_list[0])
    return [[word2id[w] for w in words] for words in words_list]
  
def make_vocab(wc):
    word2id, id2word = {}, {}
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    word2id['<sos>'] = START
    word2id['<eos>'] = END
    i = 4
    for w in wc:
      if w != '<SOS>' and w!='<EOS>':
        word2id[w] = i
        i += 1
    return word2id  
  

def make_embedding(id2word, w2v_file, emb_dim):
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    embedding = nn.Embedding(vocab_size, emb_dim).weight

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            if i == START:
                embedding[i, :] = torch.Tensor(w2v['<SOS>'])
            elif i == END:
                embedding[i, :] = torch.Tensor(w2v['<EOS>'])
            elif id2word[i] in w2v:
                embedding[i, :] = torch.Tensor(w2v[id2word[i]])
            else:
                oovs.append(i)
    return embedding, oovs  
  

def sequence_loss(logits, targets,pad_idx=-1):
    """ functional interface of SequenceLoss"""
    assert logits.size()[:-1] == targets.size()
    mask = targets != pad_idx
    target = targets.masked_select(mask)
    logit = logits.masked_select(
          mask.unsqueeze(2).expand_as(logits)
    ).contiguous().view(-1, logits.size(-1))
    loss = F.cross_entropy(logit, target, reduction='none')
    assert (not math.isnan(loss.mean().item())
              and not math.isinf(loss.mean().item()))
    return loss  

def coll_fn_extract(data):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts, num = d
        return source_sents and extracts
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch

def load_dataset(path):
    documents = []
    for filename in os.listdir(path):
        file_num = filename.split('.')[0]
        #print(join(path,filename))
        with open(join(path+"/",filename)) as f:
         js = json.loads(f.read())
        if js["extracted"] and min(js["extracted"]) < args.max_sents_article:
            documents.append((js["article"],js["extracted"],file_num)) # gli scores non ci dovrebbero servire a nulla, tanto tutti 1.0
    return documents

def load_ext_net(ext_dir,cuda):
    ext_meta = json.load(open(join(ext_dir, 'meta.json')))
    assert ext_meta['net'] == 'ml_rnn_extractor'
    ext_ckpt = load_best_ckpt(ext_dir,"cuda" if cuda else "cpu")
    ext_args = ext_meta['net_args']
    vocab = pkl.load(open(join(ext_dir, 'vocab.pkl'), 'rb'))
    ext = PtrExtractSumm(**ext_args)
    ext.load_state_dict(ext_ckpt)
    return ext, vocab

def main(args):
  cuda=True
  if args.no_cuda:
    cuda=not args.no_cuda

  
  w2v = gensim.models.Word2Vec.load(args.w2v_file).wv
  
  wc = []
  for word in w2v.vocab.items():
    wc.append(word[0])
  word2id = make_vocab(wc)
  id2word = {i: w for w, i in word2id.items()}
  vocab_size = len(word2id)
  net,_=load_ext_net(args.extractor_model,cuda)
  #net = PtrExtractSumm(emb_dim=300,vocab_size=vocab_size,conv_hidden=args.conv_hidden,lstm_hidden=args.lstm_hidden,lstm_layer=args.lstm_layer,bidirectional=args.bidirectional)
  #net.load_state_dict(torch.load(args.extractor_model))
  
  
  if cuda:
    net = net.cuda()
  net.eval()
  
  doc=load_dataset(args.dir)
  loader = DataLoader(doc, batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=coll_fn_extract)

  os.mkdir(args.output_dir)
  with torch.no_grad():
    
    for sample in loader:
      data = {}
      
      sources, extracted, file_num = tuple(map(list, unzip(sample)))
      if args.k == 0:
        k = len(extracted[0]) # if k is not specified (i.e. k defaults to 0) extract as many sentences as there are in the ground truth. Note that this is possible only if we have a ground truth
      elif args.k > len(sources[0]):
        k = len(sources[0]) # this is because otherwise if the request is to extract more sentences then there actually are in the article, it would be impossible. Simply extract all the sentences of the article in this case
      else:
        k = args.k
        
      batch = zip(sources, extracted)

      batch = prepro_fn_extract(args.max_words_article,args.max_sents_article,batch)

      batch = convert_batch_extract_ptr(UNK, word2id, batch)
      fw_args = batchify_fn_extract_ptr_dec(PAD,batch,cuda=cuda)
      article_sents, sent_nums = fw_args
      predicted = net.extract(article_sents, sent_nums, k)
      file_num = file_num[0]
      with open(args.dir +"/"+ file_num + ".json", "r") as f:
        js = json.loads(f.read())
        data['article'] = sources
        data['extracted'] = extracted
        data['predicted'] = predicted
        data['gold'] = js['gold']
        with open(join(args.output_dir, '{}.json'.format(file_num)), 'w') as f:
          json.dump(data, f, indent=4)
        print(file_num)
      
      
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='extraction of the labels from pretrained extractor model'
    )
    parser.add_argument('--dir', required=True, help='directory of the training samples')
    parser.add_argument('--no-cuda',default=False, action="store_true",help='set cuda gpu')
    # model options
    parser.add_argument('--w2v_file', action='store',
                        help='pretrained word2vec embedding')
    parser.add_argument('--extractor_model',action='store',help='pretrained extractor model')
    parser.add_argument('--conv_hidden', type=int, action='store', default=100,
                        help='the number of hidden units of Conv')
    parser.add_argument('--lstm_hidden', type=int, action='store', default=256,
                        help='the number of hidden units of LSTM')
    parser.add_argument('--lstm_layer', type=int, action='store', default=2,
                        help='the number of layers of LSTM Encoder')
    parser.add_argument('--bidirectional', action='store', default=True,
                        help='enable or disable bidirectional LSTM encoder')

    # length limit
    parser.add_argument('--max_words_article', type=int, action='store', default=100,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_sents_article', type=int, action='store', default=300,
                        help='maximun sentences in an article')
    parser.add_argument('--k', type=int, action='store', default=0,
                        help='how many sentences to extract per article. If not defined will extract as many sentences as there are in the golden summary')
    
    # training options
    parser.add_argument('--output_dir', action='store', default="./",
                        help='directory for the storage of the predicted extractions')
    parser.add_argument('--num_workers', type=int, action='store', default=2,
                        help='number of workers for the data loader')
    args = parser.parse_args()
    
    print(args)

    main(args)
