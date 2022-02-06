import sys
from gensim.models import Word2Vec
'''
from gensim.models import Word2Vec
model = Word2Vec.load("word2vec.model")
model.most_similar[""]
'''


if __name__ == "__main__":
    '''
    arg1: corpus_processed path (file,read)
    arg2: word2vec model path (file,save)
    '''
    if (len(sys.argv)==3):

        PATH_COR_PROC=sys.argv[1] #path of corpus processed
        PATH_W2V=sys.argv[2] #path where to save the w2v model
        #Read the filtered coprus in a list
        corpus_filtered=[]
        with open(PATH_COR_PROC, "r",encoding="utf8") as f:
            row=f.readline().strip()
            while (row):
                corpus_filtered.append(row.split(" "))
                row=f.readline().strip()    
        print("Number of line of the corpus:",len(corpus_filtered))

        print("\nTraining...")

        model = Word2Vec(corpus_filtered,sg=1,min_count=3,window=2,size=300,sample=6e-5,alpha=0.05,negative=20,workers=16,iter=15)
        model.save(PATH_W2V)
        
        print("Word2vec model saved at:",PATH_W2V)
    else:
        print("ERROR: you must secify the path of corpus used to train and the path where save the w2v model (c:/folder/w2v.model)")