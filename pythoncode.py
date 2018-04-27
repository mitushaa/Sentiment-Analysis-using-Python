import sys                                
import time
import csv
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pyodbc
import sqlalchemy


from sqlalchemy import create_engine
from sqlalchemy import sql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Numeric  
from sqlalchemy.orm import sessionmaker
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import wordnet
from nltk.corpus import webtext
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.stem import WordNetLemmatizer
from nltk.chunk.regexp import ChunkString, ChunkRule, ChinkRule, RegexpParser
from nltk.tree import Tree
from nltk.tag import untag
from featx import bag_of_words


#from tabulate import tabulate
#from langdetect import detect


from string import punctuation
from NBClassify import nb_classifier
from collections import Counter

#from wordcloud import WordCloud
#from scipy.misc import imread

from functools import reduce
from functools import partial

#Sys Arguments - Apart from File name , Batch ID to be passed from the web application
#print ('Number of arguments:', len(sys.argv), 'arguments.')
#print ('Argument List:', str(sys.argv))

#Classes
Base = declarative_base()
#Declaration of the class in order to write into the database. This structure is standard and should align with SQLAlchemy's doc.
class Current(Base):
    __tablename__ = 'tableName'


    id = Column(Integer, primary_key=True)
    Date = Column(String(500))
    Type = Column(String(500))
    Value = Column(Numeric())
    
    def __repr__(self):
        return "(id='%s', Date='%s', Type='%s', Value='%s')" % (self.id, self.Date, self.Type, self.Value)
    """      
    def convert_to_str(field):
        return sql.func.CONVERT(
               sql.literal_column('VARCHAR(500)'), 
               field, 
               sql.literal_column(500)
               )    
    """
    
#Functions

def get_dict(cursor,sql):
    
    cursor.execute(sql)
    #declare a dictionary
    schema =  {}
    #get column name
    colname = "tokenword"
        
    for row in cursor.fetchall():
        schema.setdefault(colname, []).append(row[-1])
      
    return (schema) 
    #close cursor

def connectionstring():
    connectionstring='DRIVER={SQL Server};SERVER=(172.16.0.20);DATABASE=NLP;UID=sa;PWD=intelenet123'
    return connectionstring
    

def database_connect():
    #returns cursor
    cnxn = pyodbc.connect(connectionstring(),autocommit=True)
    cursor = cnxn.cursor()
    return cursor

def dbcon():
    #returns connection
    cnxn = pyodbc.connect(connectionstring())
    return cnxn

def UpdateBatchFlag(batchid): # 

    #print("reached Here")
    cursor = database_connect() # Connect to database        
    sql_query = "exec USP_CompletedBatchFlag '%s'" % batchid    
    #print(sql_query)
    cursor.execute(sql_query)
    cursor.close

  
def get_list(cursor,sql):
    
    cursor.execute(sql)
    #declare a dictionary
    schema =  []
    
        
    rows = [x[0] for x in cursor.fetchall()]    
    for row in rows:
        schema.append (row)
      
    #print (schema)
    #print("/n")
    return (schema) 
  

def WordBucket(text,dictionary):
    
    cdict = {} # dictionary
    #dictionary = Pos_words
    cdict = dictionary # assign the dictionary name received ( this dictionary should be already populated)
    WordList=[] 
       
    for eachitem in text:
        #print(eachitem)
        #print("/n")
        for item in cdict.values():
            
            if eachitem in item:
                WordList.append(str(eachitem))
    
    return WordList


def print_elapsed_time(timestamp,step):
    elapsed_time = time.clock() - timestamp
    print ("Step Completed %s" %step)            
    print ("Time elapsed: {} seconds".format(elapsed_time))
    
    return time.clock()
    
    
    
    
def SentimentScore(text):


    scr1 = 0
    N = len(text)
        
    for i in range(N):
        if text[i] in PosWords.values():            
            scr1 = scr1 + 1
            if i >1:
                if text[i-1] in EmpWords:
                    scr1 = scr1 + 1
                if text[i-1] in NotWords:
                    scr1 = scr1 - 2
        elif text[i] in NegWords:
            scr1 = scr1 - 1
            if text[i] in NegHighWords:
                scr1=scr1-2
            if i > 1:
                if text[i-1] in EmpWords:
                    scr1 = scr1 - 1
                if text[i-1] in NotWords:
                    scr1 = scr1 + 2
    return scr1    

'''-----------------------------------------------------------------------------------------'''
'''----------- Cleaning Level 1 & 2 ------------------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
#Word Sets
cachedStopWords = stopwords.words("english")
web_words = ['https:','"','""','",','""!' ,'-','+','=','_','~','^','^','?','.','..','...','....','(?)', ',', '&','%','!','@','$','!!','!!!','|','||','|||','/','\\','\\.', '+','(',')','<','>','><','u','h', 'www','img','border','color','style','padding','table','font','thi','inch','ha','width','height']
spl_words1 = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','he','al','ae','cv','de','fo','ie','ma','na','op','re','xy','yz','ha','ig','id','id:']
spl_words2 = ['let','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','yes','no','yesno', 'was','the', 'or', 'me','id', 'you', 'my', 'that', 'no', 'your', 'them', 'out', 'do','msg','if', 'ur', 'to', 'a', 'the', 'and', 'is','of', 'we', 'are', 'it', 'am', 'for', 'by', 'they', 'in', 'at', 'this', 'do', 'on', 'have', 'but', 'all', 'be', 'any', 'so']
spl_punct = ['(',')',':','=','??','..','\\','//','!!','@@','##','$$','%%','&&','((','))',':)',':(',':|',':/','./','[]','}{','{}','--','++',':,',').','),','.?',',?','?.','?,','(?)','<>','><','\\.','//.','++','==','_','__','+-','**','-.','+.',')?','??.','??,',':-',':.','.:','()','//,','>:','|?','"?',',-',',,','\\','(,','-?','-:','??:','+:',',-)']
rmv_punctuations = '''!()-[]{}:='"\<>/@#$%^&*_~'''
def clean_text(x):
    if pd.isnull(x) is not True:
        x = re.sub(r'[^\x00-\x7F]','', x)         # Remove all Non ASCII characters
        x = re.sub(r'[\d]','',x)                  # Remove all numbers
        x = re.sub(r'[\n]','',x)                  # Remove New Line Char
        x = re.sub('[!@#$]', '', x)               # Remove any spl Char
        x = re.sub(r'(<)?(\w+@\w+(?:\.\w+)+)(?(1)>)','',x) # Remove Email ID
        x = re.sub(r'^\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,3}$','',x)#Remove Email
        x = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) #Remove HTML link
        x = re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE)        #Remove HTML link
        x = re.sub(r'<[^>]*>', '', x)                                 # Remove all HTML tags
        x = re.sub(r'&nbsp;','',x)                                    # Remove &nbsp tags
        x = " ".join(x.split())                   # Remove all white spaces
        x = x.lower()                             # Lower Case
        x = ' '.join([word for word in x.split() if word not in web_words])
        x = ' '.join([word for word in x.split() if word not in spl_words1])
        x = ''.join(''.join(s)[:2] for _, s in itertools.groupby(x)) # Remove repetition of letters
    return x


def clean_text4(x):                   # To clean special punctuation marks
    if pd.isnull(x) is not True:
        x = ' '.join([word for word in x.split() if word not in spl_punct])
    return x


def clean_text5(x):
    if pd.isnull(x) is not True:
        x = ' '.join([word for word in x.split() if word not in cachedStopWords]) # Remove Stop Words
        x = ' '.join([word for word in x.split() if word not in spl_words2])      # Remove Special words
    if len(x) < 1:    # added by Rajesh Rajamani to handle nulls
        return x.strip()
    else:
        return x
    
def rmv_punct(line):                    # Remove Punctuation other than ,.?
    cwrd = ""
    for wrd in line:
        if wrd not in rmv_punctuations:
            cwrd = cwrd + wrd
    return cwrd


def clean_text_1(line):                 # http, //, www etc
    cwrd = []
    for wrd in line:
        if wrd not in web_words:
            cwrd.append(wrd)
    return cwrd

def clean_text_2(line):                 # a,b,c,d etc
    cwrd = []
    for wrd in line:
        if wrd not in spl_words1:
            cwrd.append(wrd)
    return cwrd

def clean_Stp_Wrd(line):                # Stop Words
    cwrd = []
    for wrd in line:
        if wrd not in cachedStopWords:
            cwrd.append(wrd)
    return cwrd


def clean_punct(line):                  # .,;,-,+, etc
    cwrd = []
    for wrd in line:
        if wrd not in list(punctuation):
            cwrd.append(wrd)
    return cwrd


def clean_spl_punct(line):              # ///,+++,---, etc
    cwrd = []
    for wrd in line:
        if wrd not in spl_punct:
            cwrd.append(wrd)
    return cwrd


'''-----------------------------------------------------------------------------------------'''
'''---------  1. Replacing RE & Correcting Words (I've -> I have)  ---------------------------------'''
'''-----------------------------------------------------------------------------------------'''
import re
replacement_patterns = [
(r'won\'t\b', 'will not'),
(r'can\'t\b', 'can not'),
(r'i\'m\b', 'i am'),
(r'&amp;\b', 'and'),
(r'ain\'t\b', 'is not'),
(r'(\w+)\'ll\b', '\g<1> will'),
(r'(\w+)n\'t\b', '\g<1> not'),
(r'(\w+)\'ve\b', '\g<1> have'),
(r'(\w+)\'s\b', '\g<1> is'),
(r'(\w+)\'re\b', '\g<1> are'),
(r'(\w+)\'d\b', '\g<1> would')
]

test_list=[]
class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    def replace(self, text):
        s = text
        try:
        #{            
            test_list.append(s)
            for (pattern, repl) in self.patterns:
            #{
                try:
                #{
                    (s, count) = re.subn(pattern, repl, s)
                #}
                except:
                #{
                    s='exception'
                #}
            #}
                
        #}
        except:
        #{
            s='exception'
        #}
        return s


replacer1 = RegexpReplacer()

'''-----------------------------------------------------------------------------------------'''
'''---------- 2.  Repeating Char (Hellooooo -> Helloo)   ---------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
class RepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w)\3(\w*)')
        self.repl = r'\1\2\3\4'
    def replace(self, word):
        
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word

replacer2 = RepeatReplacer()

'''-----------------------------------------------------------------------------------------'''
'''----------- 3. Replace Short words (bday -> BirthDay) ---------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map
    def replace(self, word):
        return self.word_map.get(word, word)

    def replace_sent(self, sent):
        i, l = 0, len(sent)
        words = []
        while i < l:
            word = sent[i]
            word = self.replace(word)
            words.append(word)
            i += 1
        return words

class CsvWordReplacer(WordReplacer):
    def __init__(self, fname):
        word_map = {}
        for line in csv.reader(open(fname)):
            word, syn = line
            word_map[word] = syn
        super(CsvWordReplacer, self).__init__(word_map)


replacer3 = CsvWordReplacer('D:/nlpproject/includes/sms_word.csv')

'''----------------------------------------------------------------------------------'''
'''--------  Replacing negations with antonyms   -----------------------------------------'''
'''----------------------------------------------------------------------------------'''

class AntonymReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map
    def replace(self, word):
        return self.word_map.get(word, word)
    def replace_negations(self, sent):
        i, l = 0, len(sent)
        words = []
        while i < l:
            word = sent[i]
            if word == 'not' and i+1 < l:
                ant = self.replace(sent[i+1])
                if ant:
                    words.append(ant)
                    i += 2
                    continue
            words.append(word)
            i += 1
        return words


class CsvWordReplacer1(AntonymReplacer):
    def __init__(self, fname):
        word_map = {}
        for line in csv.reader(open(fname)):
            word, syn = line
            word_map[word] = syn
        super(CsvWordReplacer1, self).__init__(word_map)


replacer_ar = CsvWordReplacer1('D:/nlpproject/includes/antonyms.csv')



'''-----------------------------------------------------------------------------------------'''
'''----------     Word Lemmatizer  (Ex Believes --> Belief , Absolutely --> Absolute   -----'''
'''-----------------------------------------------------------------------------------------'''
def mywnl(se):
    wnl= WordNetLemmatizer()
    return [wnl.lemmatize(k) for k in se]

'''-----------------------------------------------------------------------------------------'''
'''----------------------  TOKENIZE    ---------------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
tockenizer=WordPunctTokenizer()

sentTokenizer = PunktSentenceTokenizer()

'''-----------------------------------------------------------------------------------------'''
'''----------------- Extracting Positive & Negative Words   ----------------------------------'''
'''-----------------------------------------------------------------------------------------'''
"""
Pos_words=open("Nlpproject\\includes\\positive-words.txt","r")
Neg_words=open("Nlpproject\\includes\\negative-words.txt","r")
Emp_words=open("Nlpproject\\includes\\emphasis-words.txt","r")
Not_words=open("Nlpproject\\includes\\negation-words.txt","r")
Neg_high_words=open("Nlpproject\\includes\\negative-high-words.txt","r")
Pos_Emotion = open("Nlpproject\\includes\\positive_emotion.txt","r")
Neg_Emotion = open("Nlpproject\\includes\\negative_emotion.txt","r")


Pos_words=Pos_words.read().split(",")
Neg_words=Neg_words.read().split(",")
Neg_high_words=Neg_high_words.read().split(",")
Emp_words=Emp_words.read().split(",")
Not_words=Not_words.read().split(",")
Pos_Emotion= Pos_Emotion.read().split(",")
Neg_Emotion= Neg_Emotion.read().split(",")

"""

#assign cursor for loading various dictionaries
cursor = database_connect() # returns cursor

#load dictionaries
Pos_words = get_dict(cursor,"select tokenword from Dictionary where positivewordflag=1 ") # to be replaced with procedure 
Neg_words = get_dict(cursor,"select tokenword from Dictionary where negativewordflag=1 ") # to be replaced with procedure 
Emp_words = get_dict(cursor,"select tokenword from Dictionary where Emphasiswordflag=1 ") # to be replaced with procedure 
Not_words = get_dict(cursor,"select tokenword from Dictionary where negationwordflag=1 ") # to be replaced with procedure 
Neg_high_words =get_dict(cursor,"select tokenword from Dictionary where negativehighwordflag=1 ") # to be replaced with procedure 
Pos_Emotion = get_dict(cursor,"select tokenword from Dictionary where negativeemotionflag=1 ") # to be replaced with procedure 
Neg_Emotion = get_dict(cursor,"select tokenword from Dictionary where positiveemotionflag=1 ") # to be replaced with procedure 

# get the dictionaries loaded as list ( as this function needs an ordering in the list for iteration)    
PosWords = get_list(cursor,"select tokenword from Dictionary where positivewordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list 
NegWords = get_list(cursor,"select tokenword from Dictionary where negativewordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list  
EmpWords = get_list(cursor,"select tokenword from Dictionary where Emphasiswordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list 
NotWords = get_list(cursor,"select tokenword from Dictionary where negationwordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list 
NegHighWords =get_list(cursor,"select tokenword from Dictionary where negativehighwordflag=1 ") # to be replaced with procedure , need to work on converting the dictionary into a list 


#close cursor
cursor.close()

"""Word Bags 
pWrd = []   # Positive Word List
nWrd = []   # Negative Word List
peWrd = []  # Positive Emotion
neWrd = []  # Negative Emotion

def posWrd(text):
    pWrd = []
    for token in text:
        if token in Pos_words:
            pWrd.append(str(token))
    return pWrd

def negWrd(text):
    nWrd = []
    for token in text:
        token=str(token)
        if token in Neg_words:
            nWrd.append(str(token))
    return nWrd


def posEmoWrd(text):
    peWrd = []
    for token in text:
        if token in Pos_Emotion:
            peWrd.append(str(token))
    return peWrd

def negEmoWrd(text):
    neWrd = []
    for token in text:
        token=str(token)
        if token in Neg_Emotion:
            neWrd.append(str(token))
    return neWrd

"""

'''-----------------------------------------------------------------------------------------'''
'''----------------- Sentiment Score --- Positive & Negative Score   -----------------------'''
'''-----------------------------------------------------------------------------------------'''

scr1 = 0
def SentiScoreNew(text):
    scr1 = 0
    N = len(text)
    for i in range(N):
        if text[i] in PosWords:
            scr1 = scr1 + 1
            if i >1:
                if text[i-1] in EmpWords:
                    scr1 = scr1 + 1
                if text[i-1] in NotWords:
                    scr1 = scr1 - 2
        elif text[i] in NegWords:
            scr1 = scr1 - 1
            if text[i] in NegHighWords:
                scr1=scr1-2
            if i > 1:
                if text[i-1] in EmpWords:
                    scr1 = scr1 - 1
                if text[i-1] in NotWords:
                    scr1 = scr1 + 2
    return scr1

scr2 = 0
def PosSentiScore(text):
    scr2 = 0
    N = len(text)
    for i in range(N):
        if text[i] in PosWords:
            scr2 = scr2 + 1
            if i >1:
                if text[i-1] in EmpWords:
                    scr2 = scr2 + 1
                if text[i-1] in NotWords:
                    scr2 = scr2 - 2
    return scr2

scr3 = 0
def NegSentiScore(text):
    scr3 = 0
    N = len(text)
    for i in range(N):
        if text[i] in NegWords:
            scr3 = scr3 - 1
            if i > 1:
                if text[i-1] in EmpWords:
                    scr3 = scr3 - 1
                if text[i-1] in NotWords:
                    scr3 = scr3 + 2
    return scr3

'''--------------------------------------------'''
'''--------    Sentiment           --------------'''
def SentimentClassifier(data_frame):
    #combines the functionality of both SentiMent and SentiConfidence
    #Accepts a data frame

    #SentiMent

    data_frame['Senti']='Neutral'
    data_frame.loc[data_frame['Score']< 0,'Senti'] ='Negative'
    data_frame.loc[data_frame['Score']> 0,'Senti'] ='Positive'

    #SentiConfidence

    data_frame['ConfidenceLevel']='error'
    data_frame.loc[data_frame['Score']< -20,'ConfidenceLevel'] ='99%'
    data_frame.loc[data_frame['Score']> -20,'ConfidenceLevel'] ='90%'
    data_frame.loc[data_frame['Score']> -10,'ConfidenceLevel'] ='70%'
    data_frame.loc[data_frame['Score']> -5,'ConfidenceLevel'] ='40%'
    data_frame.loc[data_frame['Score']> -1,'ConfidenceLevel'] ='20%'
    data_frame.loc[data_frame['Score']> 0,'ConfidenceLevel'] ='10%'
    data_frame.loc[data_frame['Score']> 2,'ConfidenceLevel'] ='30%'
    data_frame.loc[data_frame['Score']> 5,'ConfidenceLevel'] ='70%'
    data_frame.loc[data_frame['Score']> 10,'ConfidenceLevel'] ='90%'
    data_frame.loc[data_frame['Score']> 20,'ConfidenceLevel'] ='99%'


def SentiMent(score):
    if score > 0:
        return 'Positive'
    elif score <0:
        return 'Negative'
    else:
        return 'Neutral'
   '''--------------------------------------------'''
def SentiConfidence(score):
    if score >20:
        return "99%"
    elif score >10:
        return "90%"
    elif score >5:
        return "70%"
    elif score >2:
        return "30%"
    elif score >0:
        return "10%"
    elif score > -1 :
        return "20%"
    elif score > -5:
        return "40%"
    elif score > -10:
        return "70%"
    elif score > -20:
        return "90%"
    elif score <-20:
        return "99%"
    else:
        return "error"

'''-----------------------------------------------------------------------------------------'''
'''------------------- POS Tagging   --------------------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
from nltk.corpus import treebank
from nltk.tag import DefaultTagger, UnigramTagger

train_sents = treebank.tagged_sents()[:3000]

tagger1 = DefaultTagger('NN')
tagger2 = UnigramTagger(train_sents, backoff=tagger1)
'''-----------------------------------------------------------------------------------------'''
'''------------------- Chunking with POS Tagging ---------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
chunker = RegexpParser(r'''
    NP:
        {<DT>?<NN.*><VB.*><DT.*>?<NN.*>}
        {<DT>?<NN.*><IN><DT><NN.*>}
        {<NN.*><VB.*><NN.*>}
        
    ''')

chunker2 = RegexpParser(r'''
    Phrase:
        {<JJ.*><NN.*>}
        {<RB><JJ>^<NN.*>}
        {<JJ><JJ>^<NN.*>}
        {<NN.*><JJ>^<NN.*>}
        {<RB.*><VB.*>}
    ''')


chunkerPOS = RegexpParser(r'''
    Noun:
    {<PDT><DT><RB><RBR><RBS>?<WDT><DT><CD><JJ><JJR><JJS><NN><NNS><NNP><NNPS>?<NN><NNS><NNP><NNPS>?<JJ><JJR><JJS>}
        
    Verb:
        {<VB><VBD><VBG><VBN><VBP><VBZ>?<PDT><DT><RB><RBR><RBS>?<WDT><DT><CD><JJ><JJR><JJS><NN><NNS><NNP><NNPS>?<NN><NNS><NNP><NNPS>?<JJ><JJR><JJS>?<RB><RBR><RBS>}
        
    Adj:
        {<JJ.*>}
    ''')

chunkerPOS_old_backup = RegexpParser(r'''
    Noun:
{<DT.*><NN.*><IN.*><DT><NN.*>}
        {<NN.*><NN.*>?<NN.*>?<NN.*>}
        {<NN.*><IN.*><NN.*>}
        
    Verb:
        {<DT.*><VB.*><NN.*>}
        {<VB.*><NN.*>}
    Adj:
        {<JJ.*>}
    ''')


'''-------------------------------------------------------------------------------------'''
'''-------------------  Extracting Root Cause   -------------------------------------------'''
'''-------------------------------------------------------------------------------------'''
rc = []
def FeatureExtractor(tree):
    rc = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            rc.append(str(untag(subtree)))
    return rc

rc1 = []
def NounExtractor(tree):
    rc1 = []
    for subtree in tree.subtrees():
        if subtree.label() == 'Noun':
            rc1.append(str(untag(subtree)))
    return rc1

rc2 = []
def VerbExtractor(tree):
    rc2 = []
    for subtree in tree.subtrees():
        if subtree.label() == 'Verb':
            rc2.append(str(untag(subtree)))
    return rc2

rc3 = []
def AdjExtractor(tree):
    rc3 = []
    for subtree in tree.subtrees():
        if subtree.label() == 'Adj':
            rc3.append(str(untag(subtree)))
    return rc3

rc4 = []
def PhraseExtractor(tree):
    rc4 = []
    for subtree in tree.subtrees():
        if subtree.label() == 'Phrase':
            rc4.append(str(untag(subtree)))
    return rc4

'''-----------------------------------------------------------------------------------------'''
'''----------------------Customized POS Tagging -----------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
ptagger = DefaultTagger('PWD')
ntagger = DefaultTagger('NWD')
etagger = DefaultTagger('EMP')

tag_pos = ptagger.tag(PosWords) # changed to list version of Dictionary
tag_neg = ntagger.tag(NegWords) # changed to list version of Dictionary
tag_emp = etagger.tag(EmpWords) # changed to list version of Dictionary

tag_wrd = tag_pos + tag_neg + tag_emp
tag_wrd_dict = dict(tag_wrd)


tagger5 = UnigramTagger(model = tag_wrd_dict, backoff= tagger2)


'''-----------------------------------------------------------------------------------------'''
'''------------------- Chunking with POS Tagging ---------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''
chunker1 = RegexpParser(r'''
    PWD:
        {<PWD><NN.*>}
        {<PWD><JJ.*>}
        {<PWD><VB.*>}
        {<RB.*><PWD>}
        {<NN.*><PWD>}
    NWD:
        {<NWD><NN.*>}
        {<NWD><JJ.*>}
        {<NWD><VB.*>}
        {<RB.*><NWD>}
        {<NN.*><NWD>}
    ''')

'''--------------  Extracting Positive & Negative Sentence  ------------------------------------------'''

ps = []
def PosExtractor(tree):
    ps = []
    for subtree in tree.subtrees():
        if subtree.label() == 'PWD':
            ps.append(str(untag(subtree)))
    return ps

ns = []
def NegExtractor(tree):
    ns = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NWD':
            ns.append(str(untag(subtree)))
    return ns

'''-----------------------------------------------------------------------------------------'''
'''------------------- Create Corpus & Frequency Matrix ------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''

Wrd = []
def WordCorpus(text):
    for token in text:
        Wrd.append(str(token))
    return Wrd

pWrd = []
def pWordCorpus(text):
    for token in text:
        #token=str(token)
        if token in Pos_words:
            pWrd.append(str(token))
    return pWrd

nWrd = []
def nWordCorpus(text):
    for token in text:
        token=str(token)
        if token in Neg_words:
            nWrd.append(str(token))
    return nWrd


'''-----------------------------------------------------------------------------------------'''
'''------------------- Categories ----------------------------------------------------------'''
'''-----------------------------------------------------------------------------------------'''

cat = pd.read_csv('D:/nlpproject/categories/c1.csv',encoding='latin-1')
catg = pd.read_csv('D:/nlpproject/categories/c2.csv',encoding='latin-1')
#Categories 1
cat1A = cat['cat1A'].tolist()
cat1B = cat['cat1B'].tolist()
cat1C = cat['cat1C'].tolist()

#Categories 2
cat2A = cat['cat2A'].tolist()
cat2B = cat['cat2B'].tolist()
cat2C = cat['cat2C'].tolist()

#Categories 3
cat3A = cat['cat3A'].tolist()
cat3B = cat['cat3B'].tolist()
cat3C = cat['cat3C'].tolist()



def CatScore1(text):
    scr = 0
    for token in text:
        if token in cat1A:
            scr=scr+1
        if token in cat1B:
            scr=scr+0.5
        if token in cat1C:
            scr=scr-1
    return (scr)

def CatScore2(text):
    scr = 0
    for token in text:
        if token in cat2A:
            scr=scr+1
        if token in cat2B:
            scr=scr+0.5
        if token in cat2C:
            scr=scr-1
    return (scr)

def CatScore3(text):
    scr = 0
    for token in text:
        if token in cat3A:
            scr=scr+1
        if token in cat3B:
            scr=scr+0.5
        if token in cat3C:
            scr=scr-1
    return (scr)
    


'''-----------------------------------------------------------------------------------------'''
'''------------------- Main Function : Run All Functions -------------------------------'''
'''-----------------------------------------------------------------------------------------'''


if __name__ == "__main__":
   #print (str((sys.argv[1:2]) ))
   #var_batchid =sys.argv[1]


start_time = time.clock()


test = pd.read_csv('D:/nlpproject/csv.csv',encoding='utf-8')  ,encoding='ANSI'    latin-1              # Reading CSV data
#test=pd.read_excel('D:/Pankaj/Others/Delhi/Rajesh R/11-may-2017/Book2.xlsx')
#for testing "20160225113129"
#batchid = var_batchid # should be passed by web application for Python engine to process 
#sql_query = "select DateCol as date,TextCol as text,RecordID as RecordID from TextDataImport where RecordID < 1000 and BatchID = '%s'" % batchid
#test = GetRecordsForAnalysisFromSQL(batchid)

#sql_string = "exec USP_GetRecordsForAnalysis '%s'" % batchid    
#test=pd.read_sql(sql_string,dbcon())

#rename columns to suit references further 
#DateColumn >>> date , TextColumn >>> text

test.rename(columns={'DateColumn': 'date', 'TextColumn': 'text'}, inplace=True)

elapsedtime = print_elapsed_time(start_time,"Import")

'''---------------------------------------------------------------------------------'''
test['RealText'] = test['text']                             # Real Text
test['text'] = test['text'].apply(clean_text)               # Cleaning Text

elapsedtime = print_elapsed_time(elapsedtime,"Cleaning Text")

test['text'] = test['text'].apply(replacer1.replace)        # I've -> I have

elapsedtime = print_elapsed_time(elapsedtime,"Replacer 1")

test['text'] = test['text'].apply(replacer2.replace)        # Helloooo -> Helloo

elapsedtime = print_elapsed_time(elapsedtime,"Replacer 2")
test['text'] = test['text'].apply(clean_text4)              # Remove Special Punctuation

elapsedtime = print_elapsed_time(elapsedtime,"Special Punctuations1")
test['text'] = test['text'].apply(rmv_punct)                # Remove Special Punctuation

elapsedtime = print_elapsed_time(elapsedtime,"Special Punctuations2")

'''---------------------------------------------------------------------------------'''
''' Text is cleaned data but with Punctuation and preposition and stop words. Text1 is completely cleaned'''
test['text_AllCleaned'] = test['text'].apply(clean_text5)               # Final Cleaning Text
elapsedtime = print_elapsed_time(elapsedtime,"Final CLeaning")

test['Token1'] =test['text_AllCleaned'].apply(tockenizer.tokenize)      # Tokenizer
elapsedtime = print_elapsed_time(elapsedtime,"Tokenizer")

test['Token1'] =test['Token1'].apply(clean_text_1)            # Clean text level 2
elapsedtime = print_elapsed_time(elapsedtime,"CleanText Level 2")

test['Token1'] =test['Token1'].apply(clean_text_2)            # Clean Few Stop Words
elapsedtime = print_elapsed_time(elapsedtime,"CleanText StopWords")

test['Token1'] =test['Token1'].apply(clean_Stp_Wrd)           # Cleaned all English Stop Words
elapsedtime = print_elapsed_time(elapsedtime,"All English Stopwords")

test['Token1'] =test['Token1'].apply(clean_punct)             # Cleaned Punctuation
elapsedtime = print_elapsed_time(elapsedtime,"CleanPunctuation")

test['Token1'] =test['Token1'].apply(clean_spl_punct)         # Cleaned Special Punctuation
elapsedtime = print_elapsed_time(elapsedtime,"SPL_Punctuation")

'''---------------------------------------------------------------------------------'''
test['Token'] =test['text'].apply(tockenizer.tokenize)      # Tokenizer
elapsedtime = print_elapsed_time(elapsedtime,"Another tokenizer")

elapsedtime = print_elapsed_time(elapsedtime,"Special Punctuations1")
test['text'] = test['text'].apply(rmv_punct)                # Remove Special Punctuation

elapsedtime = print_elapsed_time(elapsedtime,"Special Punctuations2")

'''---------------------------------------------------------------------------------'''
''' Text is cleaned data but with Punctuation and preposition and stop words. Text1 is completely cleaned'''
test['text_AllCleaned'] = test['text'].apply(clean_text5)               # Final Cleaning Text
elapsedtime = print_elapsed_time(elapsedtime,"Final CLeaning")

test['Token1'] =test['text_AllCleaned'].apply(tockenizer.tokenize)      # Tokenizer
elapsedtime = print_elapsed_time(elapsedtime,"Tokenizer")

test['Token1'] =test['Token1'].apply(clean_text_1)            # Clean text level 2
elapsedtime = print_elapsed_time(elapsedtime,"CleanText Level 2")

test['Token1'] =test['Token1'].apply(clean_text_2)            # Clean Few Stop Words
elapsedtime = print_elapsed_time(elapsedtime,"CleanText StopWords")

test['Token1'] =test['Token1'].apply(clean_Stp_Wrd)           # Cleaned all English Stop Words
elapsedtime = print_elapsed_time(elapsedtime,"All English Stopwords")

test['Token1'] =test['Token1'].apply(clean_punct)             # Cleaned Punctuation
elapsedtime = print_elapsed_time(elapsedtime,"CleanPunctuation")

test['Token1'] =test['Token1'].apply(clean_spl_punct)         # Cleaned Special Punctuation
elapsedtime = print_elapsed_time(elapsedtime,"SPL_Punctuation")

'''---------------------------------------------------------------------------------'''
test['Token'] =test['text'].apply(tockenizer.tokenize)      # Tokenizer
elapsedtime = print_elapsed_time(elapsedtime,"Another tokenizer")



test['PosCause'] =test['Tree1'].apply(PosExtractor)         # Positive Sentence Extractor
elapsedtime = print_elapsed_time(elapsedtime,"POS Extract")

test['NegCause'] =test['Tree1'].apply(NegExtractor)         # Negative Sentence Extractor
elapsedtime = print_elapsed_time(elapsedtime,"Negative Extract")

'''---------------------------------------------------------------------------------'''
test['RootCause'] =test['Tree'].apply(FeatureExtractor)     # Feature Extractor
elapsedtime = print_elapsed_time(elapsedtime,"Feauture Extractor")

test['Phrase'] =test['Tree2'].apply(PhraseExtractor)        # Two Word Phrase Extractor
elapsedtime = print_elapsed_time(elapsedtime,"Phrase Extractor")

test['Noun'] = test['TreePOS'].apply(NounExtractor)         # Noun Extractor
elapsedtime = print_elapsed_time(elapsedtime,"Noun Extractor")

test['Verb'] = test['TreePOS'].apply(VerbExtractor)         # Verb Extractor
elapsedtime = print_elapsed_time(elapsedtime,"Verb Extractor")

test['Adj'] = test['TreePOS'].apply(AdjExtractor)           # Adjective Extractor
elapsedtime = print_elapsed_time(elapsedtime,"Adjective Extractor")

'''---------------------------------------------------------------------------------'''
