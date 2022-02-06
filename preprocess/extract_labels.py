


from data_utils.preprocess import label
import sys


PATH_REP_PROC="./processed_data/annual_reports_processed"
PATH_SUM_PROC="./processed_data/gold_summaries_processed"
PATH_LABELLED="./processed_data/pointer_data"



if __name__ == "__main__":
    '''
    arg1: report_processed path (folder,read)
    arg2: summaries processed path (folder,read)
    arg3: labelled data path (folder,save)
    '''
    if (len(sys.argv)==4):
        PATH_REP_PROC,PATH_SUM_PROC,PATH_LABELLED=sys.argv[1],sys.argv[2],sys.argv[3]
        label(PATH_REP_PROC,PATH_SUM_PROC,PATH_LABELLED)

   
    else:
        print("ERROR: you must secify the path of processed reports folder, the paths of processed summaries, and the path where to save the labelled data")

