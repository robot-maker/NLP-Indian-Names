import nltk
import random
import numpy as np
import pandas as pd

def Count_ngrams(Name):
    Name_List = list(Name[0][0])
    
    if len(Name_List) <= 3:
        Name_List.insert(0,'a')

    if Name[1][0] == 'Male':
       Unigrams_Male[Alpha_Numeric[Name_List[-1]]] = Unigrams_Male[Alpha_Numeric[Name_List[-1]]] + 1
       Bigrams_Male[(Alpha_Numeric[Name_List[-2]],Alpha_Numeric[Name_List[-1]])] = Bigrams_Male[(Alpha_Numeric[Name_List[-2]],Alpha_Numeric[Name_List[-1]])] + 1
       Trigrams_Male[(Alpha_Numeric[Name_List[-3]],Alpha_Numeric[Name_List[-2]],Alpha_Numeric[Name_List[-1]])] = Trigrams_Male[(Alpha_Numeric[Name_List[-3]],Alpha_Numeric[Name_List[-2]],Alpha_Numeric[Name_List[-1]])] + 1
       
       for i in range((len(Name_List)-2),-1,-1):
           if Name_List[i] in Vowels:
               Bigrams_Vowel_Male[(Alpha_Numeric[Name_List[i]],Alpha_Numeric[Name_List[-1]])] = Bigrams_Vowel_Male[(Alpha_Numeric[Name_List[i]],Alpha_Numeric[Name_List[-1]])] + 1
               break
        
              
    elif Name[1][0] == 'Female':
        Unigrams_Female[Alpha_Numeric[Name_List[-1]]] = Unigrams_Female[Alpha_Numeric[Name_List[-1]]] + 1
        Bigrams_Female[(Alpha_Numeric[Name_List[-2]],Alpha_Numeric[Name_List[-1]])] = Bigrams_Female[(Alpha_Numeric[Name_List[-2]],Alpha_Numeric[Name_List[-1]])] + 1
        Trigrams_Female[(Alpha_Numeric[Name_List[-3]],Alpha_Numeric[Name_List[-2]],Alpha_Numeric[Name_List[-1]])] = Trigrams_Female[(Alpha_Numeric[Name_List[-3]],Alpha_Numeric[Name_List[-2]],Alpha_Numeric[Name_List[-1]])] + 1
        
        for i in range((len(Name_List)-2),-1,-1):
           if Name_List[i] in Vowels:
               Bigrams_Vowel_Female[(Alpha_Numeric[Name_List[i]],Alpha_Numeric[Name_List[-1]])] = Bigrams_Vowel_Female[(Alpha_Numeric[Name_List[i]],Alpha_Numeric[Name_List[-1]])] + 1
               break
         

def Belief_Unigrams(Name):
    Name_List = list(Name)
    
    if Unigrams_Male[Alpha_Numeric[Name_List[-1]]] >= Unigrams_Female[Alpha_Numeric[Name_List[-1]]]:
        return 'Male'
    else:
        return 'Female'
        
def Belief_Bigrams(Name):
    Name_List = list(Name)

    if Bigrams_Male[(Alpha_Numeric[Name_List[-2]],Alpha_Numeric[Name_List[-1]])] >= Bigrams_Female[(Alpha_Numeric[Name_List[-2]],Alpha_Numeric[Name_List[-1]])]:
        return 'Male'
    else:
        return 'Female'
        
def Belief_Trigrams(Name): 
    Name_List = list(Name)
    
    if len(Name_List) <= 3:
        Name_List.insert(0,'a')
    
    if Trigrams_Male[(Alpha_Numeric[Name_List[-3]],Alpha_Numeric[Name_List[-2]],Alpha_Numeric[Name_List[-1]])] >= Trigrams_Female[(Alpha_Numeric[Name_List[-3]],Alpha_Numeric[Name_List[-2]],Alpha_Numeric[Name_List[-1]])]:
        return 'Male'
    else:
        return 'Female'
        
def Belief_Vowels(Name):
    Name_List = list(Name)

    for i in range((len(Name_List)-2),-1,-1):
        if Name_List[i] in Vowels:
            break
        
    if Bigrams_Vowel_Male[(Alpha_Numeric[Name_List[i]],Alpha_Numeric[Name_List[-1]])] >= Bigrams_Vowel_Female[(Alpha_Numeric[Name_List[i]],Alpha_Numeric[Name_List[-1]])]:
        return 'Male'
            
    else:
        return 'Female'

Vowels = 'aeiouAEIOU'
Vowels_Y = 'aeiouyAEIOUY'
Vowels_H = 'aeiouhAEIOUH'
Vowels_Y_H = 'aeiouhyAEIOUHY'

Vowels_Accuracy = 0.0
Unigrams_Accuracy = 0.0
Bigrams_Accuracy = 0.0
Trigrams_Accuracy = 0.0

Unigrams_Male = np.zeros(27)
Unigrams_Female = np.zeros(27)
Unigrams_Male = np.zeros(27)
Unigrams_Female = np.zeros(27)
Bigrams_Male = np.zeros((27,27))
Bigrams_Female = np.zeros((27,27))
Trigrams_Male = np.zeros((27,27,27))
Trigrams_Female = np.zeros((27,27,27))
Bigrams_Vowel_Male = np.zeros((27,27))
Bigrams_Vowel_Female = np.zeros((27,27))

Indian_Names= []

Alpha_Numeric = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,
                 'k':11,'l':12,'m':13,'n':14,'o':15,'p':16,'q':17,'r':18,'s':19,'t':20,
                 'u':21,'v':22,'w':23,'x':24,'y':25,'z':26,'A':1,'B':2,'C':3,'D':4,'E':5,
                 'F':6,'G':7,'H':8,'I':9,'J':10, 'K':11,'L':12,'M':13,'N':14,'O':15,
                 'P':16,'Q':17,'R':18,'S':19,'T':20,'U':21,'V':22,'W':23,'X':24,'Y':25,'Z':26}
                 
Indian_Names_Dataset = pd.read_csv("Indian_Hindu_Names.txt", header=0, delimiter="\t", quoting=3)
for i in xrange(0,len(Indian_Names_Dataset)):
   Indian_Names.append((nltk.word_tokenize(Indian_Names_Dataset['Name'][i]),nltk.word_tokenize(Indian_Names_Dataset['Gender'][i])))

Indian_Names_Dataset = pd.read_csv("Indian_Muslim_Names.txt", header=0, delimiter="\t", quoting=3)
for i in xrange(0,len(Indian_Names_Dataset)):
   Indian_Names.append((nltk.word_tokenize(Indian_Names_Dataset['Name'][i]),nltk.word_tokenize(Indian_Names_Dataset['Gender'][i])))
   
Indian_Names_Dataset = pd.read_csv("Indian_Christian_Names.txt", header=0, delimiter="\t", quoting=3)
for i in xrange(0,len(Indian_Names_Dataset)):
   Indian_Names.append((nltk.word_tokenize(Indian_Names_Dataset['Name'][i]),nltk.word_tokenize(Indian_Names_Dataset['Gender'][i])))

random.shuffle(Indian_Names)
Chunk_Size = (len(Indian_Names)/10)

for i in xrange(0,10):

    for j in xrange(0,(i*Chunk_Size)):
        Count_ngrams(Indian_Names[j])
    for j in xrange(((i+1)*Chunk_Size),(10*Chunk_Size)):
        Count_ngrams(Indian_Names[j])
    
    for k in xrange((i*Chunk_Size),((i+1)*Chunk_Size)):
        
        if Belief_Vowels(Indian_Names[k][0][0]) == Indian_Names[k][1][0]:
            Vowels_Accuracy = Vowels_Accuracy + 1
        if Belief_Unigrams(Indian_Names[k][0][0]) == Indian_Names[k][1][0]:
            Unigrams_Accuracy = Unigrams_Accuracy + 1
        if Belief_Bigrams(Indian_Names[k][0][0]) == Indian_Names[k][1][0]:
            Bigrams_Accuracy = Bigrams_Accuracy + 1
        if Belief_Trigrams(Indian_Names[k][0][0]) == Indian_Names[k][1][0]:
            Trigrams_Accuracy = Trigrams_Accuracy + 1
            
    print 'Case:',i
    print ((Vowels_Accuracy/Chunk_Size)*100)
    print ((Unigrams_Accuracy/Chunk_Size)*100)
    print ((Bigrams_Accuracy/Chunk_Size)*100)
    print ((Trigrams_Accuracy/Chunk_Size)*100)
    
    Vowels_Accuracy = 0.0
    Unigrams_Accuracy = 0.0
    Bigrams_Accuracy = 0.0
    Trigrams_Accuracy = 0.0

print len(Indian_Names)
