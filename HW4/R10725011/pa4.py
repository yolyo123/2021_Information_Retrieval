import nltk
import glob
import os
import math
import numpy as np
from nltk.stem import PorterStemmer
nltk.download('stopwords')
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
  stop_words = nltk.corpus.stopwords.words('english')
  # print(stop_words)
  result = []
  alpha = '''abcdefghijklmnopqrstuvwxyz'''
  for t in smtext:
    if not t in stop_words and not t in alpha and t != '':
      result.append(t)
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

def tfidf_unitvec_doc(d, TFtds):
  ALL_doc_unit = {}
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
    ALL_doc_unit[i+1] = [doc_index, doc_unit]
    # for k in range(len(doc_unit)):
    #   doc += str(doc_index[k]) + ' ' + str(doc_unit[k]) + '\n'
    # with open('.\\output\\doc'+ filename,'w') as f:
    #   f.write(doc)
  return ALL_doc_unit
  
def all_unitV(ALL_doc_unit):
  dlen = 0
  with open('.\\dictionary.txt','r') as f:
    for line in f:
      dlen += 1
  # ALL_doc_unit = {1: [[index], [doc unit]]}
  ALL_unitV = {}
  # print(len(ALL_doc_unit))
  for j in range(1,len(ALL_doc_unit)+1):
    Vx = [0]*dlen
    # print(ALL_doc_unit[j])
    for i in range(len(ALL_doc_unit[j][0])):
      Vx[int(ALL_doc_unit[j][0][i])-1] = float(ALL_doc_unit[j][1][i]) 

    Vx = np.asarray(Vx)
    ALL_unitV[j] = Vx
  return ALL_unitV

def cosine(Vx, Vy):
  cs = np.inner(Vx, Vy)
  # print("cosine: ", cs)
  return cs

def Kdoc(I, A):
  # print(sum(I))
  I_np = np.asarray(I)
  live = np.where(I_np == 1)[0]
  live = list(live)
  # print(live)
  cluster = {}
  for l in live: 
    cluster[l+1] = [l+1]
  # print(cluster)
  A_r = A.copy()
  A_r.reverse()
  # print(A_r)
  for a in A_r:
    # print(a)
    if a[0]-1 not in live:
      for key, vallist in cluster.items():
        if a[0] in vallist:
          cluster[key].append(a[1])
    else:
      cluster[a[0]].append(a[1])
  # print(cluster)
  kdoc = ""
  filename = str(sum(I))
  for key, item in cluster.items():
    item = sorted(item)
    for i in item:
      kdoc += str(i)+ '\n'
    kdoc += '\n'
  with open('.\\'+ filename + '.txt','w') as f:
    f.write(kdoc)

def max_heapify(P,k):
    l = left(k)
    r = right(k)
    if l < len(P) and P[l] > P[k]:
        largest = l
    else:
        largest = k
    if r < len(P) and P[r] > P[largest]:
        largest = r
    if largest != k:
        P[k], P[largest] = P[largest], P[k]
        max_heapify(P, largest)

def left(k):
    return 2 * k + 1

def right(k):
    return 2 * k + 2

def build_max_heap(P):
    n = int((len(P)//2)-1)
    for k in range(n, -1, -1):
        max_heapify(P,k)

if __name__ == '__main__':
  # Generate Dictionary
  d, TFtds = generatedic()
  # d = [['t_index', 'term', df]*12306]
  # print(d)
  # print(len(TFtds))
  print("Finish generatedic")

  # Transfer each document into a tf-idf unit vector
  ALL_doc_unit = tfidf_unitvec_doc(d, TFtds)
  # ALL_doc_unit = {docID: [[index], [doc unit]]}
  # print(ALL_doc_unit)
  print("Finish tfidf_unitvec_doc")

  # get all unitV 
  ALL_unitV = all_unitV(ALL_doc_unit)
  # ALL_unitV = {docID: [unitV]}
  # print(ALL_unitV)
  print("Finish all_unitV")

  # Simple HAT with complete link
  print("Init HAC")
  C = []
  P = []
  I = []
  A = []
  N = len(TFtds)
  K = [8, 13, 20]
  for n in range(N):
    C.append([])
    P.append([])
    # print(n+1)
    for i in range(N):
      sim = cosine(ALL_unitV[n+1], ALL_unitV[i+1])
      C[n].append([sim, i+1])

    I.append(1)
    # print(C[n])
    P[n] = C[n].copy()
    P[n].remove(C[n][n])
    build_max_heap(P[n])
    # print(P[n])
  # print(P) 
  # P: n[ i[ [1.0, 1], [0.2705026724104795, 2],...], ...]  (sorted by sim)
  # C: n[ i[ [1.0, 1], [0.2705026724104795, 2],...], ...]  
  print("Finish init HAC")
  
  print("Start HAC")
  for k in range(N-1):
    max_sim = 0
    for j in range(N):
      if I[j] == 1:
        # P[j][0] => 在P[j=docID]中sim最高的pair => P[j][0] = [sim, with docID]
        sim = P[j][0][0]
        # print(sim)
        # 一定要<=，不然最後當sim一樣時會抓不到後面的cluster
        if max_sim <= sim:
          max_sim = sim
          k1 = min((j+1), P[j][0][1])
          k2 = max((j+1), P[j][0][1])


    print("max_sim: ", max_sim)
    print(k1, k2)
    A.append([k1, k2])
    # print("A: ", A)
    I[k2-1] = 0
    # print("I: ", I)

    # K = 8, 13, 20
    if sum(I) in K:
      # print(sum(I))
      Kdoc(I, A)

    # print(P[k1-1])
    P[k1-1] = []
    # print(P[k1-1])

    for j in range(N):
      # print("JJJJJ: ", j, k1-1, I[j])
      if I[j] == 1 and j != (k1-1):
        # print("in if")
        # print("orgin P[j]: ", P[j])
        # print("C[j][k1-1]: ", C[j][k1-1])
        # print("C[j][k2-1]: ", C[j][k2-1])
        P[j].remove(C[j][k1-1])
        P[j].remove(C[j][k2-1])
        # print("remove P[j]: ", P[j])

        # Complete Linked
        newsim = min(C[j][k1-1][0], C[j][k2-1][0])
        # print("newsim: ", newsim)
        C[j][k1-1][0] = newsim
        P[j].append([newsim, k1])
        build_max_heap(P[j])
        # print("P[j]: ", j, P[j])
        C[k1-1][j][0] = newsim
        P[k1-1].append([newsim, j+1])
        build_max_heap(P[k1-1])
        # print("P[k1-1]: ", k1-1, P[k1-1])
  # print("Last A: ",A)
  print("Finish HAC")
  


  




  
  
