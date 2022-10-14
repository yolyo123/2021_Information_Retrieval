import nltk
import urllib.request
import math
import csv
import copy
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
  result = []
  for t in smtext:
    if not t in stop_words and t != '':
      result.append(t)
  return result

# Likelihood
def likelihood(n11, n10, n01, n00):
  N = n11 + n10 + n00 + n01
  pt = (n11 + n01) / N
  p1 = n11 / (n11 + n10)
  p2 = n01 / (n01 + n00)
  # print(N, pt, p1, p2, n11, n10, n01, n00)
  # -2logλ = -2log(L(H1) / L(H2))
  result = 0
  up = math.pow(pt, n11) * math.pow((1-pt), n10) * math.pow(pt, n01) * math.pow((1-pt), n00)
  down = math.pow(p1, n11) * math.pow((1-p1), n10) * math.pow(p2, n01) * math.pow((1-p2), n00)
  result = (-2) * math.log(up/down, 10)
  return result

# Extract Vocob and output top features
def ExtractVoc(D, all_term_table): 
# D is dictionary {class:{docIDs:[doc text]}} 
# all_term_table = {term: [0[pre, abs],...,12[pre, abs]]}
  # Prepare for calling Likelihood
  N = CountDocs(D)
  V = {}
  for c in range(len(D)):
    print("ExtractVoc class:", c+1)
    V[c+1] = {}
    for term in all_term_table:
      n11 = all_term_table[term][c][0]
      n10 = all_term_table[term][c][1]
      n00 = 0
      n01 = 0
      # print(len(all_term_table[term]))
      for i in range(len(all_term_table[term])):
        if i != c:
          n01 += all_term_table[term][i][0]
          n00 += all_term_table[term][i][1]
      V[c+1][term] = likelihood(n11, n10, n01, n00)
    
    # Get (500 - (500%13))/13 highest value from every classes' likelihood dic V = {class:{term:val}}
    # (500 - (500%13))/13
    feature_num_perclass = int(( 500 - ( 500 % len(all_term_table[term]) ) )/ len(all_term_table[term])) 
    # Sort by value
    V[c+1] = dict(sorted(V[c+1].items(), key=lambda item: item[1], reverse=True)[:feature_num_perclass])
  # print(V)

  # Get top_feature by merging all terms in V
  top_feature = []
  for c in range(len(V)):
    for term, val in V[c+1].items():
      top_feature.append(term)
  top_feature = list(set(top_feature))
  return  top_feature # top_feature = [term*484]

def CountDocs(D): # D is dictionary {class:{docIDs:[doc text]}}
  N = 0
  for i in range(len(D)):
    N += len(D[i+1])
  return N

def CountDocsInClass(D, c): # D is dictionary {class:{docIDs:[doc text]}}
  Nc = len(D[c])
  return Nc

# Concat all docs in a class
def ConcatTextOfAllDocsInClass(D, c): # D is dictionary {class:{docIDs:[doc text]}}
  textc = ""
  for docID, doc in D[c].items():
    textc += doc
  # print(textc)
  return textc

# Get every terms and thier tf in textc (textc is all docs in a class)
def CountTokensOfAllTerm(textc, top_feature): # top_feature = [term*484]
  tokens = tokenize(textc)
  Tct = {}
  for t in tokens:
    if t in top_feature:
      if Tct.get(t, 0) == 0:
        Tct[t] = 1
      else:
        Tct[t] += 1
  return Tct # Tct = {term:tf}

# Get D's every term and their df
def GetTermDfOfAllDocs(D): # D is dictionary {class:{docIDs:[doc text]}}
  all_terms_df = {}
  for c in range(len(D)):
    for docID, doc in D[c+1].items():
      doc_terms = list(set(tokenize(doc)))
      for term in doc_terms:
        if all_terms_df.get(term, 0) == 0:
          all_terms_df[term] = 1 
        else:
          all_terms_df[term] += 1 
  return all_terms_df # all_terms_df = {term:df}

# Get D's every term's df of present/absent in/off class 
def GetAllTermTable(D): # D is dictionary {class:{docIDs:[doc text]}}
  all_term_table = {}
  all_terms_df = GetTermDfOfAllDocs(D)
  
  for c in range(len(D)):
    print("GetAllTermTable class:", c+1)
    for term, df in all_terms_df.items():
      if all_term_table.get(term, 0) == 0:
        all_term_table[term] = []
      all_term_table[term].append([0, len(D[c+1])])
    
    for docID, doc in D[c+1].items():
      doc_terms = list(set(tokenize(doc)))
      for term in doc_terms:
        all_term_table[term][c][0] += 1 
        all_term_table[term][c][1] = len(D[c+1]) - all_term_table[term][c][0]
  return all_term_table # all_term_table = {term: [0[pre, abs],...,12[pre, abs]]}

# Train of Multinomial NB
def TrainMultinomialNB(Class, D): # D is dictionary {class:{docIDs:[doc text]}}
  all_term_table = GetAllTermTable(D) # all_term_table = {term: [c1[pre, abs],...,c13[pre, abs]]}
  top_feature = ExtractVoc(D, all_term_table) # top_feature = [term*484]
  N = CountDocs(D)
  prior = []
  Tct = {}
  condprob = [[0 for i in range(Class)] for j in range(len(top_feature))]
  # condprob = {} #{term:[class_prob*13]}
  for c in range(Class):
    print("TrainMultinomialNB class:", c+1)
    Nc = CountDocsInClass(D, c+1)
    prior.append(Nc / N)
    textc = ConcatTextOfAllDocsInClass(D, c+1)
    Tct = CountTokensOfAllTerm(textc, top_feature)
    total_tf = sum(Tct.values())
    for i in range(len(top_feature)):
      if Tct.get(top_feature[i], 0) == 0:
        condprob[i][c] = (0 + 1) / (total_tf + len(top_feature))
      else:
        condprob[i][c] = (Tct[top_feature[i]] + 1) / (total_tf + len(top_feature))
  # print("condprob: ", condprob)
  # print("prior: ", prior)
  print("Finish Training")
  return top_feature, prior, condprob

# Test of Multinomial NB
def ApplyMultinomialNB(Class, V, prior, condprob, d):
  tokens = tokenize(d)
  W = []
  for t in tokens:
    if t in V:
      W.append([t, V.index(t)])

  score = copy.deepcopy(prior) # prior = [class_val*13]
  for c in range(Class):
    score[c] = math.log(prior[c], 10)
    for i in range(len(W)):
      term_index = W[i][1]
      score[c] += math.log(condprob[term_index][c], 10)
  return score.index(max(score))+1

if __name__ == '__main__':
  # Read training set from url 
  contents = urllib.request.urlopen("https://ceiba.ntu.edu.tw/course/88ca22/content/training.txt").read()
  text_list = contents.decode('utf-8').replace('\r','').split('\n')

  Class_cnt = len(text_list)
  train_docsID = {}
  train_docsTD_l = []
  class_docID_doc = {} # docsID with classID and doc

  for i in range(Class_cnt):
    train_docsID[i] = text_list[i].split()[1:]
    class_docID_doc[i+1] = {}
    # Open every doc in the class
    for docID in train_docsID[i]:
      train_docsTD_l.append(int(docID))
      with open('.\\data\\'+ str(docID) +'.txt','r') as f:
        doc = f.read()
        class_docID_doc[i+1][docID] = doc
  

  # Training
  top_feature, prior, condprob = TrainMultinomialNB(Class_cnt, class_docID_doc)

  # Testing & Output csv
  test_docsID = list(range(1,1096))
  test_docsID = set(test_docsID).difference(train_docsTD_l)
  output = []
  print("Start Testing")
  for docID in test_docsID:
      # 對每個test doc進行testing
      # print("Test doc", docID)
      with open('.\\data\\'+ str(docID) +'.txt','r') as f:
        d = f.read()
        maxc = ApplyMultinomialNB(Class_cnt, top_feature, prior, condprob, d)
        output.append([docID, maxc])
  print("Finish All Testing")
  
  out = []
  with open('.\\output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Value'])
    for r in range(len(output)):
      writer.writerow(output[r])
      out.append(output[r][1])
    print("Finish Writing output.csv")
