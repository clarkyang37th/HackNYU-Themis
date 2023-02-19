import os
import nltk 
import string
from nltk.corpus import stopwords
from math import exp, expm1
import math 
import stop_list

stop_words = stop_list.closed_class_stop_words

class Query:
    def __init__(self,ID,query):
        self.ID = ID
        self.query = query 

    # def print_values(self):
    #     print(self.ID)
    #     print(self.query)
    
    def tokenize(self):
        #stopset = [word for word in stop_words]
        stopset, newtokens, filtered_tokens= [], [], []

        for word in stop_words:
            stopset.append(word)
        stop_punc = list(string.punctuation)
        stops = stopset+stop_punc

        tokens = nltk.wordpunct_tokenize(self.query) 
        for w in tokens:
            if w.lower() not in stops:
                newtokens.append(w)

        for b in newtokens:
            if not (b.isdigit() or b[0] == '-' and b[1:].isdigit()):
                filtered_tokens.append(b)
            
        # print(filtered_tokens == filtered_tokens_)
                
        return filtered_tokens



def tf_idf(sentence,tf_dic,idf_dic):
    vector, v = [], []     
    for token in sentence:
        # tf = 0 
        # idf = 0
        tf_ = tf_dic.get(token, 0)
        idf_ = idf_dic.get(token, 0) 
        try:
            tf = tf_dic[token]
        except: 
            tf = 0 
        try:
            idf = idf_dic[token]
        except:
            idf = 0 
        vector.append(float(tf*idf))
        v.append(float(tf_ * idf_))
    #print(v == vector)
    return vector 



#list of term frequencies dictionaries per query 
def query_tf_driver(tokenized_queries):
    tf = [] 
    for query in tokenized_queries: 
        tf.append(query_tf(query))
    return tf 

def query_tf(tokenized_string):
    #total_terms = len(tokenized_string)
    dic = {} 
    for token in tokenized_string:
        if token not in dic:
            dic[token] = 1 
        else:
            dic[token] = dic[token] + 1 

    for key in dic:
        dic[key] = float(dic[key]) 

    return dic 
#calculating term frenquenxy

def st(query, dic, length, i):
    for tok in query:
        if tok not in dic:
            dic[tok] = [0] * length 
            dic[tok][i] = 1
        else:
            dic[tok][i] = dic[tok][i] + 1
    return dic

def tok_occurency(query_tokens, is_abstract=False):
    number_docs = len(query_tokens)
    dic = {}
    for i, query in enumerate(query_tokens):
        if not is_abstract:
            dic = st(query, dic, number_docs, i)
        else:
            for sentence in query:
                dic = st(sentence, dic, number_docs, i)

    return dic

def idf(query_dic,total_docs):
    dic = {}
    for key in query_dic:
        number_docs = 0 
        for count in query_dic[key]:
            if count > 0:
                number_docs = number_docs + 1 

        
        dic[key] = math.log(float(total_docs) / float(number_docs))

    return dic 




def abstract_docs(filename):
    f = open(filename,"r").readlines()
    abstracts = [] 
    string = ""
    cont = False
    for line in f: 
        if ".I" in line: 
            cont = False 
            if len(string)>0:
                abstracts.append(string)
                string = ""
        if ".W" in line:
            cont = True 
        if cont == True:
            string = string + line 
    if len(string)>0:
        abstracts.append(string)

    new_abstracts = []
    for abst in abstracts:
        abst = abst[2:]
        new_abstracts.append(abst)
    return new_abstracts 


# def tokenized_queries(query_docs):
#     tokenized_que = []
#     for query in query_docs:
#         tokenized_que.append(query.tokenize()) 
#     return tokenized_que

def query_docs(filename):
    f = open(filename,"r").readlines()

    IDS = [] 
    queries = [] 
    WS = [] 
    cont = False 
    string = ""
    for line in f:
        #print(line)
        if ".I" in line:
            cont = False
            if len(string) > 0:
                queries.append(string)
            string = "" 
            part = line.split()
            IDS.append(part[1]) 
        if ".W" in line:
            cont = True 
        if cont == True:
            string = string + line 

    if len(string) > 0 :
        queries.append(string)

    new_queries = []

    for query in queries:
        query = query[2:]
        new_queries.append(query)


    query_docs = [] #query objects 
    length = len(IDS)
    for count in range(length):
        I = IDS[count] 
        qu = new_queries[count]
        query_docs.append(Query(I,qu)) 
    return query_docs 


def tokenize_abstract(abstract_list):
    
    abstract_sentences = [] 
    for abstract in abstract_list:
        sentences = nltk.sent_tokenize(abstract) 
        abstract_sentences.append(sentences)

    
    return abstract_sentences

def abstract_tokens(abstract_list):
    sentence_tokens = [nltk.sent_tokenize(x) for x in abstract_list]
    abstract_token = [] 
    for doc in sentence_tokens:
        stopset,toks = [],[] 
        for sentence in doc:
            for word in stop_words:
                stopset.append(word)
            #stopset = [word for word in stopwords.words('english')]
            stop_punc = list(string.punctuation)
            stops = stopset+stop_punc

            tokens = nltk.wordpunct_tokenize(sentence) 

            tokens = [w for w in tokens if w.lower() not in stops ] 

            filtered_tokens = [x for x in tokens if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]

            toks.append(filtered_tokens) 
        abstract_token.append(toks) 



    return abstract_token 


def abstracts_occurence(abstract_tokens):
    total_docs = len(abstract_tokens)
    dic = {} 
    count = 0 
    for doc in abstract_tokens:
        for sentence in doc:
            for tok in sentence: 
                if tok not in dic:
                    dic[tok] = [0]*total_docs 
                    dic[tok][count] = 1 
                else:
                    dic[tok][count] = dic[tok][count] + 1 
        count = count +1 

    return dic 

def abstract_idf(abstract_occurence_dic,total_docs):
    dic = {}

    for key in abstract_occurence_dic:
        number_docs= 0 
        for count in abstract_occurence_dic[key]:
            if count > 0:
                number_docs = number_docs + 1

        #print(number_docs)

        dic[key] = float(total_docs / number_docs )
        dic[key] = math.log(dic[key])

    return dic 

def abstract_tf(abstract_occurence_dic,total_docs):
    abstract_tf_list = [] 
    for count in range(total_docs):
        dic = {} 
        for key in abstract_occurence_dic:
            term = abstract_occurence_dic[key][count]
            if term > 0:
                dic[key] = term 
        abstract_tf_list.append(dic)
    return abstract_tf_list 


def cos_sim(vect1,vect2):
    numerator = 0
    sum_of_squares1 = 0
    sum_of_squares2 =0
    for index in range(len(vect1)):
        numerator = numerator + vect1[index]*vect2[index]
        sum_of_squares1 = sum_of_squares1 + math.pow(vect1[index],2) 
        sum_of_squares2 = sum_of_squares2 + math.pow(vect2[index],2) 
    
    
    sum_of_squares1 = math.sqrt(sum_of_squares1)
    sum_of_squares2 = math.sqrt(sum_of_squares2)
    denominator = float(sum_of_squares1 * sum_of_squares2) 
    divide = 0
    try:
        divide = float(numerator/denominator)
    except: 
        divide = 0 

    return divide

#生成scorelist
def scoring(queries,abstract_tf_list,abstract_idf,query_tf_list,query_idf):
    
    score_array = [] 
    countQ = 0
    for query in queries: 
        #find query vector 
        v1 = tf_idf(query,query_tf_list[countQ],query_idf)
        #print "v1 \t\t " ,v1 
        countA = 0
        score_tups = []
        for abstract in abstract_tf_list:
            v2 = tf_idf(query,abstract_tf_list[countA],abstract_idf)
            #print "v2 \t\t " ,cv2
            #print v1, v2
            #cosine_sim = cosine_similarity(v1,v2) 
            cosine_sim = cos_sim(v1,v2)
            out = (countQ+1,countA+1,round(cosine_sim, 12))
            score_tups.append(out)
            #print(cosine_sim)
            #cprint(v2)
            countA = countA+1 
        countQ = countQ + 1 
        score_array.append(score_tups)
    return score_array 


def score_sort(score_array):
    array = [] 
    for query in score_array:
        sorted_by_similarity = sorted(query, key=lambda tup: tup[2],reverse = True )
        array.extend(sorted_by_similarity)
    return array 


# from collections import defaultdict
# dic = defaultdict(list)
# {key: value}  {key: list()}
# dic[a].append()
# 菜鸟教程
# data是一个dict
# {"file1": [score], "file2": [score]}
# for k, v in data.items():
#   
def out_write(path, data):
    #f = open(path, "w",  newline='\n')
    with open(path, "w" ,  newline='\n') as f:
        i = 0
        k = 1
        for query_id, abstract_id, sim in data:
            if (query_id == k):
                if i< 100:
                    f.write("{} {} {}\n".format(query_id, abstract_id, sim))
                    i += 1
                else:
                    k += 1
                    i = 0





def main():
    
    for f in os.listdir(os.path.join("data")):#读取相对路径的这个文件夹中所有文件名
        if 'labeled' in f.split('.'):
            query_docs_list = query_docs("paragraph/labeled")
        elif 'dataset' in f.split('.'):
            abstract_doc_list = abstract_docs("paragraph/dataset")
        # elif 'crantest' in f.split('.'):
        #     abstract_doc_list = abstract_docs("cran/crantest.qry")
    # print("1")
    # query_docs_list = query_docs("cran/cran.qry") 
    #print("2")
    tokenized_queries_list = [x.tokenize() for x in query_docs_list]
    #tokenized_queries_list = tokenized_queries(query_docs_list) 
    #print("3")
    query_dic = tok_occurency(tokenized_queries_list)
    #print("4")
    queries_idf = idf(query_dic,len(query_docs_list))
    #print("5")
    query_tf_list = query_tf_driver(tokenized_queries_list)
    # print("6")
    # abstract_doc_list = abstract_docs("cran/cran.all.1400")
    #print(abstract_doc_list)
    #print(len(abstract_doc_list))
    #print("7")
    #abst_sentence_tokens = tokenize_abstract(abstract_doc_list)
    abst_tokens = abstract_tokens(abstract_doc_list)
    #print("8")
    abstract_tok_occurence = abstracts_occurence(abst_tokens)
    abst_idf = abstract_idf(abstract_tok_occurence,len(abstract_doc_list))
    #print("9") 
    abst_docs_tf_list = abstract_tf(abstract_tok_occurence,len(abstract_doc_list))
    #print("10" )
    score_list = scoring(tokenized_queries_list,abst_docs_tf_list,abst_idf,query_tf_list,queries_idf)
    score_list = score_sort(score_list)
    #print(score_list) 
    #print("11")
    print(score_list)
    out_write("output.txt", score_list)




main()  

#\map 分数问题，\r
