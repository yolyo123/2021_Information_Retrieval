import urllib.request
import nltk
from nltk.stem import PorterStemmer

# Read text
contents = urllib.request.urlopen("https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt").read()
text = str(contents,'utf-8')

# Remove punctuations in text
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

for e in text:
  if e in punc:
    text = text.replace(e, '')
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
# print(sptext)
# print(smtext)

# Remove Stopwords
# Using stopwords list from nltk 
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
# print(stop_words)
result = ''
sw = []
for t in smtext:
  if not t in stop_words:
    result = result + t + ' '
  else:
    sw.append(t)
print(result)
# print(sw)

# Output result txt
with open('.\\result.txt','w') as f:
  f.write(str(result)) 
