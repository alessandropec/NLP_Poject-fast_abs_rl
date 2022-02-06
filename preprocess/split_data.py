import os
import shutil
from sklearn.model_selection import train_test_split
import sys






if __name__ == "__main__":
    '''
    Starting from arg1 path generate two folder train and val
    divide the file in arg1 path in train and val with percentage of training specified

    arg1: pointer_data path (folder,read)
    arg2: pointer data splitted path (folder,save)
    arg3: train_size value (0.7) (flaot,read)
    arg4: (optional) set a random_state for splitting (default 42) (int,read)
    '''
    r_state=42
    if (len(sys.argv)==5):
        r_state=int(sys.argv[4])
    if(len(sys.argv)==5 or len(sys.argv)==4):
            
        PATH_POINTER=sys.argv[1]
        PATH_POINTER_SPLIT=sys.argv[2]
        train_size=float(sys.argv[3])
        file_names=os.listdir(PATH_POINTER)

        os.mkdir(PATH_POINTER_SPLIT)
        os.mkdir(PATH_POINTER_SPLIT+"/train")
        os.mkdir(PATH_POINTER_SPLIT+"/val")

        train,test,_,_=train_test_split(file_names,file_names,train_size=train_size,random_state=r_state)
        
        print(len(train),len(test))
        for f_name in train:
            shutil.copyfile(PATH_POINTER+"/"+f_name,PATH_POINTER_SPLIT+"/train/"+f_name)
        for f_name in test:
            shutil.copyfile(PATH_POINTER+"/"+f_name,PATH_POINTER_SPLIT+"/val/"+f_name)
   
    else:
        print("ERROR: you must secify the path of pointer data folder, and the percentage of training file (ex 0.7)")

