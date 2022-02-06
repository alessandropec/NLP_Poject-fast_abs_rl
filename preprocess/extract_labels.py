

import os
from preprocess import label
import sys

if __name__ == "__main__":
    '''
    arg1: report_processed path (folder,read)
    arg2: summaries processed path (folder,read)
    arg3: labelled data path (folder,save)
    '''
    if (len(sys.argv)==4):
        PATH_REP_PROC,PATH_SUM_PROC,PATH_LABELLED=sys.argv[1],sys.argv[2],sys.argv[3]
        os.mkdir(PATH_LABELLED)
        label(PATH_REP_PROC,PATH_SUM_PROC,PATH_LABELLED)

   
    else:
        print("ERROR: you must secify the path of processed reports folder, the paths of processed summaries, and the path where to save the labelled data")

