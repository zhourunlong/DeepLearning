from io import TextIOBase
import json
import jieba
from tqdm import tqdm
import os, sys

def filter(str):
    l = len(str)
    output = "".join(str)
    for i in range(l):
        t = ord(str[i])
        if t < 33 or (127 <= t and t < 13312) or t >= 40870:
            output = output.replace(str[i], '')
    return output

sys.path.append(os.path.dirname(__file__))

cur_dir = os.path.dirname(os.path.abspath(__file__))

stopwordFile = os.path.join(os.path.dirname(cur_dir), "Datasets/CLS/stop_words.txt")
wordLabelFile = os.path.join(os.path.dirname(cur_dir), "Datasets/CLS/vocab.txt")

worddict = {}
len_dic = {}
stoplist = open(stopwordFile, 'r', encoding='utf_8').read().split('\n')
data_num = 0

#for file_name in ["train", "dev"]:
for file_name in ["train"]:
    file = os.path.join(os.path.dirname(cur_dir), "Datasets/CLS/{}.json".format(file_name))
    with open(file, encoding="utf-8") as f:
        data = json.load(f)
    for article in data:
        texts = [article["Content"]]
        for question in article['Questions']:
            texts.append(question['Question'])
            choices = question['Choices']
            for choice in choices:
                texts.append(choice)
        
        for text in texts:
            seg = jieba.cut(text, cut_all=False)
            for w in seg:
                if w in stoplist:
                    continue
                if w in worddict:
                    worddict[w] += 1
                else:
                    worddict[w] = 1
        
        data_num += len(texts)

wordlist = sorted(worddict.items(), key=lambda item:item[1], reverse=True)
f = open(wordLabelFile, 'w', encoding='utf_8')
for t in wordlist:
    s = filter(t[0])
    if s == "":
        continue
    d = s + ' ' + str(t[1]) + '\n'
    f.write(d)
