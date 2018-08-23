
# coding: utf-8

# In[1]:


#CS 6320 - Natural Language Processing
#Team: Tokenizers; Authors: Yankai Jia, Mansi Kukreja

"""Task 1: Create a corpus of 50 FAQs and Answers
   Task 2: Implement a shallow NLP pipeline and bag-of-words matching algorithm
            o Bag-of-words creation
                 Tokenize the FAQs and Answers into bag-of-words
                 Create a bag-of-words for each FAQ
                 Tokenize the user’s input natural language question/statement into a bagof-words
            o Bag-of-words matching
                 Return the FAQ and Answer whose bag-of-words best statistically matches the bag-of-words from the user’s input natural language question/statement
            o Evaluate the results of at least 10 user questions/statements for the top-10 returned FAQ matches
    Task 3: Implement a deeper NLP pipeline to extract semantically rich features from the FAQs and Answers 
            o Tokenize the FAQs and Answers into sentences and words 
            o Remove stop-words 
            o Lemmatize the words to extract lemmas as features 
            o Stem the words to extract stemmed words as features """

import nltk
import os
#java_path = "C:/Program Files/Java/jdk1.8.0_171/bin/java.exe"
#os.environ['JAVAHOME'] = java_path
from nltk.tokenize import *
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from collections import defaultdict
import pandas as pd
from nltk.tokenize import *
from nltk import *
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.parse.stanford import StanfordDependencyParser
import operator
from functools import reduce


# In[2]:


#--------------------TASK1-------------------#
#Task 1: Create a corpus of 50 FAQs and Answers 


# In[3]:


#--------------------TASK2-------------------#


# In[4]:


#reading the file and generating two separate lists
def get_QA_list(file_path):
    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
        text = f.read().split('\n\n')
    # getting every second item of the sequence starting from 0
    question_list = text[::2]
    # getting every second item of the sequence starting from 1
    answer_list = text[1::2]
    return question_list, answer_list


# In[5]:


def combine_lists_to_dict(tuple1, list2):
    combined_dict = dict(zip(tuple1, list2))
    return combined_dict


# In[6]:


def combine_list_to_list(list1,list2):
    combined_list = list(zip(list1, list2))
    return combined_list


# In[7]:


#combine a dict's list of key and list of value into one list of list
def combine_dict_to_list(dict):
    # combined bag of QA as word
    combined_pair = []
    # combined list of QA as word
    combined_list = []
    for pair in dict:
        for lists in pair:
            combined_list = combined_list + list(lists)
        combined_pair.append(combined_list)
        combined_list = []
    return combined_pair


# In[8]:


#tokenizing list of sentences into list of words
def tokenize_into_word(list):
    list_of_words = []
    for element in list:
        tokenized_group = nltk.word_tokenize(element)
        list_of_words.append(tokenized_group)
    return list_of_words


# In[9]:


#matching algorithm for task-2 over combined pair and user Input
def maching(combined_pair, user_words):
    # statitically matching
    match_num = 0
    matched_num_dict = {}
    i = 1

    for combined_QA in combined_pair:
        for user_word in user_words:
            if (user_word.lower() in [x.lower() for x in combined_QA]):
                match_num = match_num + 1
        matched_num_dict[i] = match_num
        match_num = 0
        i = i + 1
    matched_words_dict = {}
    matched_words_list = []
    j = 1
    for combined_QA in combined_pair:
        for user_word in user_words:
            if (user_word.lower() in [x.lower() for x in combined_QA]):
                matched_words_list.append(user_word)
        matched_words_dict[j] = matched_words_list
        matched_words_list = []
        j = j + 1
    return matched_num_dict, matched_words_dict


# In[10]:


#fetching the top 10 matched faqs
def get_top10_mached(matched_words_dict, combined_pair, user_input):
    top_best_match = dict(sorted(matched_words_dict.items(), key=lambda item: len(item[1]), reverse=True))
    #taking first 10 values
    top_10_pairs = {k: top_best_match[k] for k in list(top_best_match)[:10]}
    combined_pair_dict = {}
    combined_pair_dict = dict(zip((x for x in range(1, len(combined_pair))), combined_pair))
    faq_matched_list=[]
    for key in top_10_pairs:
        if key in combined_pair_dict:
            faq_matched_list.append(combined_pair_dict.get(key))
    return top_10_pairs,faq_matched_list


# In[11]:


#creating dataframe
def create_df(top_10_pairs,faq_matched_list):
    pd.set_option('display.max_colwidth', -1)
    dataframe=pd.DataFrame(data={'FAQ No': list(top_10_pairs.keys()),
                 'FAQ & Answer': list(faq_matched_list),
                  'Number of Words Matched': list(map(len, top_10_pairs.values())),
                  'Words Matched': list(top_10_pairs.values())
                             })
    display(dataframe)


# In[12]:


#--------------------TASK3-------------------#


# In[13]:


#removing stop words and words< 3 letters
def clean_text(tokenized_pair):
    stoplist = set(stopwords.words('english'))
    clean = []
    for word in tokenized_pair:
        if word not in stoplist:
            clean.append(word)
    tokens = []
    for word in clean:
        if word not in tokens:
            if len(word) >= 3:
                tokens.append(word)
    return tokens


# In[14]:


#lemmatizing stop words list
def lemmatize(clean_list):
    # lower capitalization
    clean_list = [word.lower() for word in clean_list]
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_list=[]
    for word in clean_list:
        lemma_list.append(wordnet_lemmatizer.lemmatize(word))
    return lemma_list


# In[15]:


#Stemming the lemmatized list
def stemming(lemma_list):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in lemma_list]
    return stemmed_tokens


# In[16]:


#POS tagging
def pos_tag(tokenized_pair_list):
    postaggedwords=[]
    for word in tokenized_pair_list:
        pos_tag=nltk.pos_tag(word)
        postaggedwords.append(pos_tag)
    return postaggedwords


# In[17]:


#dependency parsing
def parser(sent,path_to_jar,path_to_models_jar):
    dependency_parser = StanfordDependencyParser(path_to_jar = path_to_jar, path_to_models_jar = path_to_models_jar)
    result = dependency_parser.raw_parse(sent)
    dep = result.__next__()
    return list(dep.triples()) 


# In[18]:


#extracting synsets of all the words to get wornet features
def synsets(word):
    list_of_synsets = wn.synsets(word)
    return list_of_synsets


# In[19]:


#extracting hypernyms
def get_hypernyms(words):
    list_of_hypernyms = []
    for word in words:
        word_synsets = synsets(word)
        for synset in word_synsets:
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    if lemma.name() not in list_of_hypernyms:
                        list_of_hypernyms.append(lemma.name())
    return (list_of_hypernyms)


# In[20]:


#extracting hyponyms
def get_hyponyms(words):
    list_of_hyponyms = []
    for word in words:
        word_synsets = synsets(word)
        for synset in word_synsets:
            for hyponyms in synset.hyponyms():
                for lemma in hyponyms.lemmas():
                    if lemma.name() not in list_of_hyponyms:
                        list_of_hyponyms.append(lemma.name())
    return (list_of_hyponyms)


# In[21]:


#extracting meronyms
def get_meronyms(words):
    list_of_meronyms = []
    for word in words:
        word_synsets = synsets(word)
        for synset in word_synsets:
            for meronyms in synset.part_meronyms():
                for lemma in meronyms.lemmas():
                    if lemma.name() not in list_of_meronyms:
                        list_of_meronyms.append(lemma.name())
    return (list_of_meronyms)


# In[22]:


#extracting holonyms
def get_holonyms(words):
    list_of_holonyms = []
    for word in words:
        word_synsets = synsets(word)
        for synset in word_synsets:
            for holonyms in synset.part_holonyms():
                for lemma in holonyms.lemmas():
                    if lemma.name() not in list_of_holonyms:
                        list_of_holonyms.append(lemma.name())
    return (list_of_holonyms)


# In[23]:


#--------------------TASK4--------------------#


# In[24]:


#matching using all the extracted features
def matching_task4(mergerd_list, user_words):
    # statitically matching
    match_num = 0
    matched_num_dict = {}
    i = 1

    for item in mergerd_list:
        for user_word in user_words:
            if (user_word in item):
                match_num = match_num + 1
        matched_num_dict[i] = match_num
        match_num = 0
        i = i + 1
    matched_words_dict = {}
    matched_words_list = []
    j = 1
    for item in mergerd_list:
        for user_word in user_words:
            if (user_word in item):
                matched_words_list.append(user_word)
        matched_words_dict[j] = matched_words_list
        matched_words_list = []
        j = j + 1
    return matched_num_dict, matched_words_dict


# In[25]:


#--------------------TEST CASES--------------------#


# In[26]:


def get_match(slist):
    print("---Naive Matching---")
    for item in slist:
        matched_num_dict, matched_words_dict = maching(combined_pair, nltk.word_tokenize(item))
        top_10_pairs,faq_matched_list=get_top10_mached(matched_words_dict,combined_pair, item)
        print(item)
        create_df(top_10_pairs,faq_matched_list)

def get_match_feature(slist):
    print("---Feature Matching---")
    for item in slist:
        matched_num_dict, matched_words_dict = matching_task4(mergerd_list, nltk.word_tokenize(item))
        top_10_pairs,faq_matched_list=get_top10_mached(matched_words_dict,mergerd_list, item)
        print(item)
        create_df(top_10_pairs,faq_matched_list) 


# In[27]:


print("----------Taking Corpus and generating tokenized list of lits of FAQs and Answers----------")
print("Enter path of the Corpus:")
file_path = input()
question_list, answer_list = get_QA_list(file_path)
#C:\Users\mansi\OneDrive - The University of Texas at Dallas\UTD(Sem2)\NLP\Project\Corpus.txt


# In[28]:


question_list_tuple=tuple(question_list)
bag_of_QA= combine_lists_to_dict(question_list_tuple, answer_list)


# In[29]:


# tokenized question list of lists
question_tokenized = tokenize_into_word(bag_of_QA)
# tokenized answer list of lists
values_bag=bag_of_QA.values()
answers_tokenized = tokenize_into_word(values_bag)


# In[30]:


# combining the list of Q as word and list of A as word into a list
tokenized_pairs=combine_list_to_list(question_tokenized,answers_tokenized)


# In[31]:


# combine list of Q and A as words into bag of QA as word
combined_pair=combine_dict_to_list(tokenized_pairs)


# In[32]:


# taking user input
print("Please enter your Question: ")
user_input = input()
user_words = nltk.word_tokenize(user_input)


# In[33]:


print("----------TASK2 RESULTS---------")
print("User Input:")
print(user_input)
# get matching information
matched_num_dict, matched_words_dict = maching(combined_pair, user_words)
# get top 10 matching Q&A
top_10_pairs,faq_matched_list=get_top10_mached(matched_words_dict,combined_pair, user_input)
create_df(top_10_pairs,faq_matched_list)


# In[34]:


print("---------TASK3---------")


# In[35]:


#list after removing stop words
clean_list = []
for clist in combined_pair:
    newlist=clean_text(clist)
    clean_list.append(newlist)
#print(clean_list)


# In[36]:


lemmatized_list=[]
for llist in clean_list:
    newlist=lemmatize(llist)
    lemmatized_list.append(newlist)
#print(lemmatized_list)


# In[37]:


stem_list = []
for clist in clean_list:
    newlist=stemming(clist)
    stem_list.append(newlist)
#print(stem_list)


# In[38]:


pos_tagged_list=pos_tag(lemmatized_list)
#print(pos_tagged_list)


# In[39]:


#dependency parsing
print("Enter Path to parser.jar:\n")
path_to_jar= input()
print("Enter Path to models.jar:\n")
path_to_models_jar = input()

#C:\Users\mansi\OneDrive - The University of Texas at Dallas\UTD(Sem2)\NLP\Project\stanford-parser-full-2018-02-27\stanford-parser.jar
#C:\Users\mansi\OneDrive - The University of Texas at Dallas\UTD(Sem2)\NLP\Project\stanford-parser-full-2018-02-27\stanford-parser-3.9.1-models.jar


# In[ ]:


print("----------Parsed Structures of Questions----------")
parsed_question_list=[]
for sentence in question_list:
    parsed_question_list.append(parser(sentence,path_to_jar,path_to_models_jar))
print(parsed_question_list)


# In[ ]:


print("----------Parsed Structures of Answers----------")
parsed_answer_list=[]
for sentence in answer_list:
    parsed_answer_list.append(parser(sentence,path_to_jar,path_to_models_jar))
print(parsed_answer_list)


# In[40]:


hyp_list=[]
for sent in lemmatized_list:
    hyp_list.append(get_hypernyms(sent))
#print(hyp_list)


# In[41]:


hyponyms_list=[]
for sent in lemmatized_list:
    hyponyms_list.append(get_hyponyms(sent))
#print(hyponyms_list)


# In[42]:


meronyms_list=[]
for sent in lemmatized_list:
    meronyms_list.append(get_meronyms(sent))
#print(meronyms_list)


# In[43]:


holonyms_list=[]
for sent in lemmatized_list:
    holonyms_list.append(get_holonyms(sent))
#print(holonyms_list)
        


# In[44]:


print("----------TASK3 RESULTS---------")
print('-----------Creating Feature Dataframe of All FAQs and Answers----------')

pd.set_option('display.max_colwidth', -1)
dataframe=pd.DataFrame({
                'Lemma List': lemmatized_list,
                  'Hypernyms': hyp_list,
                'Hyponyms': hyponyms_list,
                'Meronyms': meronyms_list,
                'Holonyms': holonyms_list,
                'POS_Tagged List':pos_tag(lemmatized_list)})

display(dataframe)
        


# In[45]:


#Merging all extracted features to list of lists
biglist=dataframe.values.tolist()
result=[]
mergerd_list=[]
#print(biglist)
for items in biglist:
    result=reduce(operator.concat, items)
    mergerd_list.append(result)
    #print(mergerd_list)


# In[55]:


#Matching users input to extracted features and getting top 10 matched
print("----------TASK4---------")
print("Enter user input:")
ui=input()
user_input_words=nltk.word_tokenize(ui)
matched_num_dict, matched_words_dict = matching_task4(mergerd_list, user_input_words)
top_10_pairs,faq_matched_list=get_top10_mached(matched_words_dict,mergerd_list, ui)
create_df(top_10_pairs,faq_matched_list)


# In[57]:




print("User Input")
print(nltk.word_tokenize(ui))
print("Parse Structure")
print(parser(ui,path_to_jar,path_to_models_jar))
print("POS Tagged List")
print(pos_tag(ui))
print("Lemma List")
print(lemmatize(nltk.word_tokenize(ui)))
print("Stemmed List")
print(stemming(lemmatize(nltk.word_tokenize(ui))))
print("Hypernyms")
print(get_hypernyms(lemmatize(nltk.word_tokenize(ui))))
print("Hyponyms")
print(get_hyponyms(lemmatize(nltk.word_tokenize(ui))))
print("Meronyms")
print(get_meronyms(lemmatize(nltk.word_tokenize(ui))))
print("Holonyms")
print(get_holonyms(lemmatize(nltk.word_tokenize(ui))))

