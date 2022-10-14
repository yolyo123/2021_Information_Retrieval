import nltk
import glob
import os
import math
import numpy as np
from nltk.stem import PorterStemmer
def tokenize(text):
  # Remove punctuations in text
  punc = '''!()-[]{};'"\,<>?@#$%^&*_`~'''
  urlpunc = '''/.:'''
  num = '''1234567890'''

  for e in text:
    if e in punc or e in urlpunc or e in num:
      text = text.replace(e, ' ')

  # Remove \r\n
  text = text.replace('\r','').replace('\n','')

  # Lowercasing
  text = text.lower()

  # Tokenization
  sptext = text.split(' ')

  # Stemming using Porter’s algorithm
  ps = PorterStemmer()
  smtext = []
  for t in sptext:
    smtext.append(ps.stem(t))

  # Remove Stopwords
  # Using stopwords list from nltk 
  nltk.download('stopwords')
  stop_words = nltk.corpus.stopwords.words('english')
  # print(stop_words)

  smstop = []
  for s in stop_words:
    for i in s:
      if i in punc:
        s = s.replace(i, '')
    smstop.append(ps.stem(s))

  result = []
  sw = []
  alpha = '''abcdefghijklmnopqrstuvwxyz'''
  for t in smtext:
    if not t in stop_words and not t in smstop and not t in alpha and t != '':
      result.append(t)
    else:
      sw.append(t)
  # print(sw)
  # print(len(result))
  return result

def generatedic():
  text = ""
  dic = {}
  TFtds = {}

  def key_func(x):
    txt = os.path.split(x)[1]
    num = int(txt.split('.')[0])
    return num
  # Read text, Tokenize, Get tf and df of the doc
  for filename in sorted(glob.glob('.\\data\\*.txt'),key=key_func ):
    # print(os.path.split(filename)[1])
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
      text = f.read()
      # Tokenize doc
      result = tokenize(text)
      # Get tf of the doc
      TFtds[os.path.split(filename)[1]] = {i:result.count(i) for i in result}
      # Get df of the doc
      result = set(result)
      result = list(result)

      for t in result:
        if dic.get(t, 0) == 0:
          dic[t] = 1
        else:
          dic[t] += 1

  # Sorted dictionary by terms
  dics = sorted(dic.items(), key=lambda x:x[0])

  # Generate dictionary.txt and return dictionary list d, TFtds back
  dictionary = ""
  d = []
  cnt = 0
  for i in dics:
    # print(i[1])
    cnt += 1
    c = str(cnt)
    i1 = str(i[1])
    dictionary += c + " " + i[0] + " " + i1 + "\n"
    d.append([c,i[0],i[1]])
  # print(dictionary)
  # Output dictionary txt
  with open('.\\dictionary.txt','w') as f:
    f.write(dictionary) 
  return d, TFtds

def cosine(Docx, Docy):
  x = []
  y = []
  dlen = 0
  with open('.\\dictionary.txt','r') as f:
    for line in f:
      dlen += 1
  # print(dlen)
  def readfile(Doc):
    Dlist = []
    with open('.\\output\\doc' + str(Doc) + '.txt','r') as f:
      for line in f:
        if '\n' in line:
          line = line.replace('\n', '')
          line = line.split(' ')
        Dlist.append(line)
      return Dlist

  x = readfile(Docx)
  y = readfile(Docy)

  Vx = [0]*dlen
  Vy = [0]*dlen
  for i in range(1, len(x)):
    Vx[int(x[i][0])-1] = float(x[i][1]) 

  for i in range(1, len(y)):
    Vy[int(y[i][0])-1] = float(y[i][1]) 

  Vx = np.asarray(Vx)
  Vy = np.asarray(Vy)

  cs = np.inner(Vx, Vy)
  # print(cs)
  return cs


if __name__ == '__main__':
  # Generate Dictionary
  d, TFtds = generatedic()
  print("Finish generatedic\n")

  # Transfer each document into a tf-idf unit vector
  for i in range(len(TFtds)):
    doc_unit = []
    doc_index = []
    doc = ""
    cnt = 0
    filename = str(i+1)+'.txt'
    TFtd = TFtds[filename]
    for j in range(len(d)):
      t_index = d[j][0]
      term = d[j][1]
      df = d[j][2]
      N = len(d)
      if term in TFtd:
        cnt += 1
        tf = TFtd[term]
        idf = math.log(N/df, 10)
        tf_idf = tf * idf
        doc_index.append(t_index)
        doc_unit.append(tf_idf)

    # print(filename, doc)
    doc = str(cnt) + '\n' + doc
    doc_unit = doc_unit / np.linalg.norm(doc_unit)

    for k in range(len(doc_unit)):
      doc += str(doc_index[k]) + ' ' + str(doc_unit[k]) + '\n'
    with open('.\\output\\doc'+ filename,'w') as f:
      f.write(doc)
  
  # Cosine similarity.
  cs = cosine(1, 2)
  print("Cosine similarity of doc1 and doc2: ", cs)

  




  
  
