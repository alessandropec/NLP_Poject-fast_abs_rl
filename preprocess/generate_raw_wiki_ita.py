import pickle
import shutil
import os
path="./italian.pkl"
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

PATH_TO_RAW="./raw_data2k"

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