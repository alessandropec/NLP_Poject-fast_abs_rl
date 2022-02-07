from preprocess import create_tokenized_corpus
from preprocess import create_bow
from preprocess import process_docs
from preprocess import filter_doc
import json
import os
import sys

HELP_STRING="\
    arg1: phase of processing->\n\
        help: print help documentation\n\
        set_up: generate the folder for containing all data usefull to preprocessing\n\
        \n\
        all: set_up and preprocess all data -> \
            arg2: report_raw path (folder, read); \n\
            arg3: summaries_raw path (folder, read);\n\
            \n\
        tok_corpus: create the corpus that containing all tokenized sent from all reports ->\n\
            arg2: report_raw path (folder, read);\n\
            arg3: corpus_tokenized path (file, save);\n\
\n\
        create_bow: from tokenized corpus generate bow and the bow of common token->\n\
            arg2: corpus_tokenized path (file, read);\n\
            arg3: bow path (folder, save)\n\
\n\
        filter_corpus: with common_bow filter out from a tokenized doc the uncommon token\n\
            arg2: corpus_tokenized path (file,read)\n\
            arg3: corpus_processed path (file,save)\n\
            arg4: common_bow path (file,read)\n\
\n\
        process_docs: tokenized and filter all file in a folder (for report and summaries)\n\
            arg2: files_raw path (folder,read)\n\
            arg3: files_processed path (folder,save)\n\
            arg4: common_bow path (file,read)"
    
    
PATH_REP_RAW="./raw_training/annual_reports"
PATH_SUM_RAW="./raw_training/gold_summaries"






def generate_dir(root):
    PATH_BOW=root+"/bow"
    PATH_COR_PROC=root+"/corpus_processed"
    PATH_REP_PROC=root+"/annual_reports_processed"
    PATH_SUM_PROC=root+"/gold_summaries_processed"
    try:
        os.mkdir(root)
    except OSError as exc:
        print(exc)
        exit()
    os.mkdir(PATH_BOW)
    os.mkdir(PATH_COR_PROC)
    os.mkdir(PATH_REP_PROC)
    os.mkdir(PATH_SUM_PROC)
    return PATH_COR_PROC,PATH_REP_PROC,PATH_SUM_PROC,PATH_BOW




def full_pipeline(PATH_REP_RAW,PATH_SUM_RAW,PATH_COR_PROC,PATH_REP_PROC,PATH_SUM_PROC,PATH_BOW,PATH_ALL_DOCS=None):

    '''**********************Create and tokenize full corpus*****************************************'''
    if PATH_ALL_DOCS!=None:
        create_tokenized_corpus(PATH_ALL_DOCS,PATH_COR_PROC,file_name="corpus_tokenized.txt")
    else:
        create_tokenized_corpus(PATH_REP_RAW,PATH_COR_PROC,file_name="corpus_tokenized.txt")

    input("The corpus was generated, press any key to continue with bag of word generation.")

    '''**********************Create BoW for filtering process*****************************************'''

    bow,common_bow=create_bow(PATH_COR_PROC+"/corpus_tokenized.txt",filter_first=20000)
    #Save complete BoW
    a_file = open(PATH_BOW+"/bow.json", "w")
    json.dump(bow, a_file)
    a_file.close()

    #Save complete BoW
    a_file = open(PATH_BOW+"/common_bow.json", "w")
    json.dump(common_bow, a_file)
    a_file.close()
    del bow
    input("The bag of word was generated and saved, press any key to continue with filtering whole corpus.")

    '''*************************Filter out uncommon token from the whole corpus****************'''

    #Note process docs process all file in the folder to process the hole corpus only this must be in the folder
    filter_doc(PATH_COR_PROC+"/corpus_tokenized.txt",PATH_COR_PROC+"/corpus_proc.txt",common_bow)

    input("The corpus have been filtered, press any key to process all reports")

    '''**********************Filter each report and summaries*****************************************'''

    process_docs(PATH_REP_RAW,PATH_REP_PROC,common_bow)

    input("All reports have been processed, press any key to process the gold summaries")

    process_docs(PATH_SUM_RAW,PATH_SUM_PROC,common_bow)

 


if __name__ == "__main__":
    '''
    arg1: phase of processing->
        help: print help documentation

        set_up: generate the folder for containing all data usefull to preprocessing
            arg2: path of folder to save the processed_data the folder must not exists (folder,save)
        
        all: set_up and preprocess all data -> 
            arg2: report_raw path (folder, read); 
            arg3: summaries_raw path (folder, read);
            arg4: path of folder to save the processed_data the folder must not exists (folder,save)
            arg5: all_docs_raw path (optional) (folder, read)

        tok_corpus: create the corpus that containing all tokenized sent from all reports ->
            arg2: report_raw path (folder, read);
            arg3: corpus_processed path (folder, save);

        create_bow: from tokenized corpus generate bow and the bow of common token->
            arg2: corpus_tokenized path (file, read);
            arg3: bow path (folder, save)

        filter_doc: with common_bow filter out from a tokenized doc the uncommon token
            arg2: corpus_tokenized path (file,read)
            arg3: corpus_processed path (file,save)
            arg4: common_bow path (file,read)

        process_docs: tokenized and filter all file in a folder (for report and summaries)
            arg2: files_raw path (folder,read)
            arg3: files_processed path (folder,save)
            arg4: common_bow path (file,read)
    
    '''
    if(len(sys.argv)<2):
        print(HELP_STRING)
        
    elif sys.argv[1]=="set_up":
        generate_dir(sys.argv[2])

    elif sys.argv[1]=="help":
        print(HELP_STRING)

    elif sys.argv[1]=="all":
        print(len(sys.argv))
        
        if len(sys.argv)==5:
            args=generate_dir(sys.argv[4])
            full_pipeline(sys.argv[2],sys.argv[3],*args)
        elif len(sys.argv)==6:
            args=generate_dir(sys.argv[4])
            full_pipeline(sys.argv[2],sys.argv[3],*args,sys.argv[5])
        else:
            print("ERROR: you must specify the correct number of arguments type help to doc")
            exit()
        print("Data have been processed look inside:",sys.argv[4])

    elif sys.argv[1]=="tok_corpus":
        if len(sys.argv)!=4:
            print("ERROR: you must specify the correct number of arguments type help to doc")
        else:
            create_tokenized_corpus(sys.argv[2],sys.argv[3],file_name="corpus_tokenized.txt")
    
    elif sys.argv[1]=="create_bow":
        if len(sys.argv)!=4:
            print("ERROR: you must specify the correct number of arguments type help to doc")
        else:            
            bow,common_bow=create_bow(sys.argv[2],filter_first=20000)
            #Save complete BoW
            a_file = open(sys.argv[3]+"/bow.json", "w")
            json.dump(bow, a_file)
            a_file.close()

            #Save complete BoW
            a_file = open(sys.argv[3]+"/common_bow.json", "w")
            json.dump(common_bow, a_file)
            a_file.close()

    elif sys.argv[1]=="filter_doc":
        if len(sys.argv)!=5:
            print("ERROR: you must specify the correct number of arguments type help to doc")
        else:
            #Read top 20000 common BoW
            a_file = open(sys.argv[4], "r")
            common_bow = json.loads(a_file.read())
            print("Number of common token:",len(common_bow))

            filter_doc(sys.argv[2],sys.argv[3],common_bow)            
    
    elif sys.argv[1]=="process_docs":
        if len(sys.argv)!=5:
            print("ERROR: you must specify the correct number of arguments type help to doc")
        else:
            #Read top 20000 common BoW
            a_file = open(sys.argv[4], "r")
            common_bow = json.loads(a_file.read())
            print("Number of common token:",len(common_bow))

            process_docs(sys.argv[2],sys.argv[3],common_bow)
            
    else:
        print("ERROR: you must specify a correct command use help to doc")
         

