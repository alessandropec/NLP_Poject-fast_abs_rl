

import sys


import os
from os.path import join
import json
import nltk
#from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer


nltk.download('punkt')
from cytoolz import compose
word_tokenize = RegexpTokenizer(r'\w+')

def create_tokenized_corpus(PATH_REP_RAW,PATH_COR_PROC,file_name="corpus_tokenized.txt",SOS="<SOS>",EOS="<EOS>"):
    '''
    Read all file liste in PATH_REP, divide by sentence (nltk.sent_tokenize)
    tokenize each sentence, filter non alphanumeric data, normalization to lower case
    add SOS and EOS token.
    Append each sentence in each document to a single file, save it in PATH_COR_TOK

    :param str PATH_REP: path of folder in which there are the reports ex c://myfolder
    :param str PATH_COR_TOK: path of file where save all sent for all report tokenized 
    :return: void
    '''

    #retrieve file name
    rep_files_name=[v for v in os.listdir(PATH_REP_RAW) ]

    print(f"Number of annual report: {len(rep_files_name)}")

    with open(PATH_COR_PROC+"/"+file_name,"w",encoding="utf8") as f:
    #Read all report inside PATH_REP
        for i in range(0,len(rep_files_name)):
    
            #Read all file as a single string in order to tokenize eah sent
            with open(PATH_REP_RAW+"/"+rep_files_name[i],"r",encoding="utf8") as fr:
                text="".join([ line.strip()+" " for line in fr.readlines()])
                rows=nltk.sent_tokenize(text)

        
            for r in rows:
                tokenized=word_tokenize.tokenize(r) #tokenization
        
                tokenized=[w.lower() for w in tokenized if w.isalnum()] #get only alphanumeric 
                
                tokenized.insert(0,SOS)#Insert special token
                tokenized.append(EOS)

                if len(tokenized)!=0:#check if the sent is not void
                    f.write(" ".join(tokenized)+"\n") #add each tokenized sent as a line in corpus file
            print(f"Added file {rep_files_name[i]} at index: {i}") 


def create_bow(PATH_COR_TOK,filter_first=20000):
    '''
    Read the corpus tokenized, generate a bag of word from this and a bag of word
    of filter_first num of token (the more common)

    :param str PATH_COR_TOK: path of the corpus tokenized file
    :param int filter_first: number of token ordered by frequence to extract
    :return tuple: nltk.FreqDist obj, bow contain all token common_bow the first filter_first (bow,common_bow)
    '''
    #Read the generated file and add all token to a single list
    flat_doc=[]
    with open(PATH_COR_TOK,"r",encoding="utf8") as f:
        row=f.readline().strip()
        while(row):    
            for t in row.split(" "):
                flat_doc.append(t)
            row=f.readline().strip()
       
    bow=nltk.FreqDist(flat_doc)
    common_bow=dict(bow.most_common(20000)) #potenzialmente i token speciali potrebbero nn esserci ma è praticamente impossibile
    print("Number of words:",len(flat_doc),"Number of distinct words:",len(bow),"Number of most common:",len(common_bow))
    return (bow,common_bow)



def filter_doc(PATH_DOC,PATH_DOC_FIL,common_bow,EOS="<EOS>",SOS="<SOS>"):
    corpus_filtered=[]
    with open(PATH_DOC,"r",encoding="utf8") as f:
        row=f.readline().strip()
        while(row):
            filtered_row=[t for t in row.split(" ") if t in common_bow.keys() or t in [EOS,SOS]]

            if len(filtered_row)!=0:
                corpus_filtered.append(filtered_row)
            row=f.readline().strip()

    with open(PATH_DOC_FIL,"w",encoding="utf8") as f:
        for row in corpus_filtered:
            f.write(" ".join(row)+"\n")   
    return corpus_filtered


def process_docs(PATH_DOCS,PATH_PROC_DATA,common_bow,SOS="<SOS>",EOS="<EOS>",suffix="_prepro"):
    '''
    Read all file from PATH_DOCS, process each file in tokenized sents, add special token
    and filter out token not present in common_bow

    :param str PATH_DOCS: path of folder containing txt file to process
    :param str PATH_PROC_DATA: path of folder that will contain the preprocess version
    :param str suffix: suffix to insert between file name and file extenions (.txt)
    '''


    rep_files_name=[v for v in os.listdir(PATH_DOCS)]
    print(f"Number of file to process: {len(rep_files_name)}")
    print("Number of common token:",len(common_bow))

    i=0
    for nameRep in rep_files_name: #Read each file
        with open(PATH_DOCS+"/"+nameRep,"r",encoding="utf8") as f:
            text="".join([ line.strip()+" " for line in f.readlines()])
            rows=nltk.sent_tokenize(text) #Divide in sentence
        with open(PATH_PROC_DATA+"/"+nameRep.split(".")[0]+suffix+".txt","w",encoding="utf8") as f: #Create new processed version o file
            for r in rows: #Tokenize each sentence
                tokenized=word_tokenize.tokenize(r) 
                tokenized=[w.lower() for w in tokenized if w.isalnum() and w.lower() in common_bow.keys() ] #Normalization and filter token not present in common_bow
                if len(tokenized)!=0: #eos and sos#Check for empty sent append special token
                    tokenized.insert(0,SOS)
                    tokenized.append(EOS)
                    f.write(" ".join(tokenized)+"\n")#Erite the processed sentence in the file
        print(f"Added file {nameRep} index:{i}") #Total: {len(rep_files_name)}")
        i+=1



'''''''''''''''''''''Function for extract label for pointer net training'''
def _split_words(texts):
    return map(lambda t: t.split(), texts)

def _lcs_dp(a, b):
    """ compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b)+1)]
          for _ in range(0, len(a)+1)]
    # dp[i][j]: lcs_len(a[:i], b[:j])
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp

def _lcs_len(a, b):
    """ compute the length of longest common subsequence between a and b"""
    dp = _lcs_dp(a, b)
    return dp[-1][-1]

def compute_rouge_l(output, reference, mode='f'):
    """ compute ROUGE-L for a single pair of summary and reference
    output, reference are list of words
    """
    assert mode in list('fpr')  # F-1, precision, recall
    lcs = _lcs_len(output, reference)
    if lcs == 0:
        score = 0.0
    else:
        precision = lcs / len(output)
        recall = lcs / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    return score

def get_extract_label(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores = []
    indices = list(range(len(art_sents)))
    for abst in abs_sents:
        rouges = list(map(lambda x: compute_rouge_l(x,reference=abst, mode='r'),art_sents))
        if not indices:
            break        
        ext = max(indices, key=lambda i: rouges[i])
        indices.remove(ext)
        extracted.append(ext)
        scores.append(rouges[ext])
    return extracted, scores

def label(PATH_REP_PROC,PATH_SUM_PROC,PATH_LABELLED):

  data = {}
  n_data = len(os.listdir(PATH_REP_PROC))
  i = 0
  try:
    os.mkdir(PATH_LABELLED)
  except OSError as exc:
      print(exc)
      return
  for filename in os.listdir(PATH_REP_PROC):
      print(filename)
      i += 1
      file_num = filename.split('_')[0]
      #print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100*i/n_data),end='')
      with open(PATH_REP_PROC + '/' + filename,encoding="utf8") as f:
        data['article'] = f.readlines()
      with open(PATH_SUM_PROC + '/' + file_num + "_1_prepro.txt",encoding="utf8") as f:
        data['gold'] = f.readlines()
      #json_data = json.dumps(data)
      tokenize = compose(list, _split_words)
      art_sents = tokenize(data['article'])
      abs_sents = tokenize(data['gold'])
      extracted, scores = get_extract_label(art_sents, abs_sents)
      data['extracted'] = extracted
      data['score'] = scores
     
      with open(join(PATH_LABELLED, '{}.json'.format(file_num)), 'w') as f:
          json.dump(data, f, indent=4)
      print(i)
