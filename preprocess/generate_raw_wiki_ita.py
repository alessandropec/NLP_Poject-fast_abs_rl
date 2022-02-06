import pickle
import shutil
import os
import sys


if __name__ == "__main__":

  max_doc=None;
  if len(sys.argv)!=3:
    if len(sys.argv)==4:
      max_doc=int(sys.argv[3])
    else:
      print("ERROR: you must specify pkl dataset path and the folder path to store the output")
      exit()
  path=sys.argv[1]
  PATH_TO_RAW=sys.argv[2]
  with open(path, 'rb') as f:
      ita_docs=pickle.load(f)

  print("Number of grouped docs:",len(ita_docs))
  listed_docs=[]
  i=0
  for docs in [list(v.items()) for k,v in ita_docs.items()]:
      for doc in docs:
          doc=list(doc)
          doc[0]=str(i)
          doc[1].pop("english_url")
          doc[1].pop("english_section_name")
          listed_docs.append(doc)
          i+=1

  print("Number of all docs",len(listed_docs))

  #print(listed_docs[0][1]["document"])
  #print(listed_docs[1][1]["document"])

  #print(listed_docs[0][1]["summary"])
  #print(listed_docs[1][1]["summary"])
  os.mkdir(PATH_TO_RAW)
  os.mkdir(PATH_TO_RAW+"/all")
  os.mkdir(PATH_TO_RAW+"/documents")
  os.mkdir(PATH_TO_RAW+"/gold_summaries")
  if max_doc!=None:
    listed_docs=listed_docs[0:max_doc]
  for doc in listed_docs[0:2000]:
      print("Saving doc and sum: ",doc[0])
      with open(PATH_TO_RAW+"/documents/"+doc[0]+".txt","w",encoding="utf8") as f:
          f.write(doc[1]["document"])
      with open(PATH_TO_RAW+"/gold_summaries/"+doc[0]+"_1.txt","w",encoding="utf8") as f:
          f.write(doc[1]["summary"])
  print("Copying file in \"all\" folder")
  for f_name in os.listdir(PATH_TO_RAW+"/documents"):
      shutil.copyfile(PATH_TO_RAW+"/documents/"+f_name,PATH_TO_RAW+"/all/"+f_name)
  for f_name in os.listdir(PATH_TO_RAW+"/gold_summaries"):
      shutil.copyfile(PATH_TO_RAW+"/gold_summaries/"+f_name,PATH_TO_RAW+"/all/"+f_name)