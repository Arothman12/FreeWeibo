#!/usr/bin/env python
# coding: utf-8

# In[185]:

def WEBIO():
# Read in the data

    import pandas as pd
    import re
    import streamlit as st

    
    df1 = pd.read_excel("C:\\Users\\avida\\freeweibo-09-15-2022.xlsx")
    #df.index.names = ['row_id']


    df2 = df1[['User_name', 'time_created', 'OriginalPostLink', 'HotTerm', 'content']]
    df2 = df2.dropna() # If column value is empty drop the entire row
    df2['orig_freeweibo_row_id'] = df2.index

    weibo_id_list = []

    for ind in df2.index:
        row_content = df2['OriginalPostLink'][ind]
        weibo_id = re.findall(r'\d+', str(row_content)) or ["Error"]
    #    print(weibo_id[0])
        weibo_id_list.append(weibo_id[0])
        
    #print(weibo_id_list)
    df2['weibo_ids'] = weibo_id_list

    # Dropping rows that contain ['0'] as a weibo id
    ###df2 = df2[df2["weibo_ids"].str.contains("0") == False]
    # Remove rows where content is 'NaN'

    # df2.head()
    print("Number of rows in original data: ", len(df1))
    print("Number of rows after removing empty rows in columns:", len(df2))
    print("Number rows removed:", len(df1) - len(df2))
    print("Range of the index:", len(df2[df2['content'] != 2].index))
    # df2['content'] != 2 subsets the df2, and len(df2.index) returns the length of the index.
    # https://stackoverflow.com/questions/47539511/how-to-get-range-of-index-of-pandas-dataframe

    df2.head()


    # In[187]:


    # Remove rows where content is 'NaN'
    # df = df.dropna()  drop rows that contain 'NaN' in any column
    new_df = df2.dropna(subset=['content'])  # drop rows that contain 'NaN' in the content column

    # Hash the content column to check for any duplicate weibo posts
    new_df['hash'] = new_df['content'].apply(hash)


    # Selecting duplicate rows in first occurrence based on the hash column
    duplicate = new_df[new_df.hash.duplicated()]
     
    print("Duplicate Rows :")
     
    # Print the duplicates
    duplicate
    #duplicate['content']
    dups_retained = duplicate['hash'].unique()  # We remove all dups except the first ocurrence
    num_dups_retained = len(dups_retained)

    #print(new_df.loc[[62884]])
    #print(new_df.loc[[62886]])

    # Drop duplicates

    new_df = new_df.drop_duplicates(subset=['hash'], keep='first')
    #print(new_df.loc[[62884]])
    #print(new_df.loc[[62886]])

    print("Numner of duplicates retained:", num_dups_retained)  # We remove all dups except the first ocurrence
    print("Number of rows after removing duplicate rows:", len(new_df))
    print("Number dupliate rows removed:", len(duplicate) - num_dups_retained)

    #print(new_df[62880:62889])

    ##print(new_df.where(new_df['hash'] < 2))
    print("Range of the index:", len(new_df[new_df['content'] != 2].index))
    #print(new_df.iloc[62113])
    ### print(new_df[new_df['orig_freeweibo_row_id'] == 62884])
    new_df.head()


    # In[188]:


    # To remove the characters from the row instead of the entire row we can run:
    import re

    new_df['content'] = new_df['content'].str.replace('@', '')              # remove @ from each row
    new_df['content'] = new_df['content'].str.replace('//@', '')            # remove //@ from each row
    new_df['content'] = new_df['content'].str.replace('\u200b', '')         # remove \u200b from each row
    new_df['content'] = new_df['content'].str.replace('?', '')              # remove ? from each row
    new_df['content'] = new_df['content'].str.replace('!', '')              # remove ! from each row
    new_df['content'] = new_df['content'].str.replace(': //', '')           # remove ! from each row

    #new_df['content'] = re.sub(r'http\S+', '', my_string)
    #new_df['content'] = new_df['content'].replace('[http\S.]', '',regex=True)

    url_pattern = r'(https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}[-a-zA-Z0-9()@:%_+.~#?&/=]*)' 
    new_df['urls'] = new_df['content'].str.extract(url_pattern, expand=False) # extract any urls and put them in a new column
    new_df['content'] = new_df['content'].str.replace(url_pattern, '')     # remove urls from the content

    new_df['content'] = new_df['content'].str.replace(' ', '')              # remove whitespace from each row
    new_df['content'] = new_df['content'].str.replace('', '')               # remove whitespace from each row

    # special characters
    special_char = re.compile('[、《【】VS.！→→？～。：，//,@_!#$%^&*()<>?/\|}{~:]')
    #new_df['symbols'] = new_df['content'].str.extract(special_char, expand=False)
    new_df['content'] = new_df['content'].str.replace(special_char,'')

    #new_df['content'] = new_df['content'].str.replace('http://', '')        # remove http:// from each row
    #new_df['content'] = new_df['content'].apply(lambda x: re.split('http:\/\/.*', str(x))[0])
    #new_df['content'] = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
                               
    filter1_df = new_df[new_df["content"].str.contains(r'[\u4e00-\u9FFF]')==True]  # Keep only Chinese characters

    filter1_df


    # In[189]:


    # Segment the cleaned content.  This will be necessary to do any pos tagging

    import pandas as pd
    import re
    import jieba
    import pinyin
    import unicodedata
     
    row_num = []
    segmented_content_list = []

    # special characters
    #special_char = re.compile('[//@_!#$%^&*()<>?/\|}{~:]')
        
    for i in filter1_df.index:
        post = filter1_df['content'][i]
        word_list = jieba.lcut(post,cut_all=True)
    #    for char in word_list:
    #        if(special_char.search(char) == None):
    #            print('String does not contain any special characters.')
    #        else:
    #            print('The string contains special characters.')
    #            print(char)
    #            symbols.append(char)
    #            segmented_content_list.append(char)
        row_num.append(i)
        segmented_content_list.append(word_list)
        
    for i in range(10):
        print(segmented_content_list[i])

    # In[190]:


    word_df = pd.DataFrame({'post_num': row_num, 'words': segmented_content_list})
    word_df.head()

    print(word_df['words'][0])
    x = word_df['words'][0]

    print(" ".join(x))

    # In[191]:


    ################################# Prepare words in each post for the word2vec model #####################
    # We need to take all the words out of each list in the word_df dataframe and save them to a file
    # with each line the file containing just the words from each post

    #     ONLY NEED TO RUN THIS ONCE SINCE IT SAVES THE OUTPUT TO A TEXT FILE freeweibo_words.txt

    ###########################################################################################################

    import pandas as pd
    import io
    import re

    # this would be the output file 
    f = io.open('freeweibo_words.txt', mode = 'w', encoding= 'utf-8')
    # word_df contains all the words for each post

    for i in range(len(word_df)):
        post = " ".join(word_df['words'][i]) + '\n'
        f.write(post)
    f.close()


    # In[192]:


    # Using the Gensim Word2Vec Model with the Chinese news dataset
    # https://www.kaggle.com/datasets/noxmoon/chinese-official-daily-news-since-2016?resource=download

    #        ONLY NEED TO RUNS THIS ONCE SINCE THE MODEL, word2.model IS SAVED TO A FILE
    ####################################################################################################
    import io
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
    import multiprocessing

    a = io.open("freeweibo_words.txt", mode="r+", encoding="utf-8")
    model = Word2Vec(LineSentence(a), vector_size=400, window = 5, min_count= 5, workers= multiprocessing.cpu_count() - 4)
    model.save('word2.model')
    model.wv.save_word2vec_format('word2vec.vector', binary=False)


    # In[199]:


    #Create model vocabulary

    import gensim
    import random

    model = gensim.models.KeyedVectors.load_word2vec_format('word2vec.vector', binary=False)
    print(type(model))

    # What is the index location for 人?
    print(model.key_to_index["人"])

    # How many words in the vocabulary?
    print(len(model))
    print(random.choice(model.index_to_key))
    model_vocab = []
    # # Print all the words in the vocabulary for the model
    for i in range(len(model)):
    #    print(model.index_to_key[i])
        model_vocab.append(model.index_to_key[i])
    print(len(model_vocab))


    # In[200]:


    ######################################################################################################

    #                        This code takes between 35-45 minutes to run

    ######################################################################################################

    import pandas as pd

    # We extract all the freeweibo posts that were used to create our model. freeweibo_posts.txt
    with open('freeweibo_words.txt', encoding="utf8") as f:
        freeweibo_posts = f.readlines()

    post_list = []
    final_weibos = []  # This list contains the words that exist in our model for each freeweibo post 
    oov = []           # oov = out of vocabulary words


    # The final_weibos post contains one list with all the posts.  We want each post to be stored in its own list
    # The for loop converts a list of all posts into individual lists for each post.
    # Code modified from https://www.geeksforgeeks.org/read-a-file-line-by-line-in-python/

    count = 0
    # Strips the newline character
    for line in freeweibo_posts:
        count += 1
    #    print("Line{}: {}".format(count, line.strip()))
        post_list.append(line.strip())

    #post_list_sample = post_list[0:10]
    post_list_sample = post_list
    ###############################################################################################################
    #             Extract all words in the model vocabulary and store in final_weibos list 
    ###############################################################################################################
    for i in range(len(post_list_sample)):
        final_weibos.append([])
        for j in range(len(post_list_sample[i])):
            word = post_list_sample[i][j]
            if word in model_vocab:
    #            print(word)
                final_weibos[i].append(word)
    #            print("Exists")
    #            print(word, "(",i,",",j,")")
            else:
    #            print(word)
                final_weibos[i].append('*')

    #            print("Doesnt Exist")
    #            print(word, "(",i,",",j,")")
    ###############################################################################################################

    ###############################################################################################################
    #               Extract all words not in the model vocabulary and store in oov list 
    ###############################################################################################################

    for i in range(len(post_list_sample)):
        oov.append([])
        for j in range(len(post_list_sample[i])):
            word = post_list_sample[i][j]
            if word in model_vocab:
    #            print(word)
                oov[i].append('*')
    #            print("Exists")
    #            print(word, "(",i,",",j,")")
            else:
    #            print(word)
                oov[i].append(word)
    #            print("Doesnt Exist")
    #            print(word, "(",i,",",j,")")

    ###############################################################################################################

    # Remove all occurrences of "*" in the final_weibos and oov lists
    #for i in range(len(final_weibos)):
        final_weibos[i] = list(filter(('*').__ne__, final_weibos[i]))

        
    #for i in range(len(oov)):
    #    oov[i] = list(filter(('*').__ne__, oov[i]))
        
    # Print all words from each post that are in the model vocabulary
    #for i in range(len(final_weibos)):
    #    print(final_weibos[i])
        
    '''
    # Print all words from each post that are not in the model vocabulary
    #for i in range(len(oov)):
    #    print(oov[i])
    '''


    # In[201]:


    # Applying the model.  

    # The last step would be to use the model to do something. As it turns out Gensim lets you do 
    # a lot of things with your model. Below are some examples Python

    import gensim

    model = gensim.models.KeyedVectors.load_word2vec_format('word2vec.vector', binary=False)

    #How many words does our model have? We simply need to check the size of our vocabulary with the Python len function:
    #print(len(model))

    # What is the upper bound of the number of weibos? We use this upper bound number in our input statement on line 51
    max_num_weibos = len(post_list_sample)

    #print the word vector associated with the word
    #print(model["中国"])
    #print(model["人"])  # Most frequently used noun in the Freeweibo dataset

    # print top 10 most similar words to the input word
    #print(model.most_similar(positive='广东', negative=None, topn=10, restrict_vocab=None, indexer=None))
    #print(model.most_similar(positive='人', negative=None, topn=10, restrict_vocab=None, indexer=None))

    # compare similarities between sentences 
    #weibo1 = final_weibos[0]
    #print(len(weibo1))

    #weibo2 = final_weibos[1]
    #print(len(weibo2))


    #weibo3 = final_weibos[2]
    #print(len(weibo3))

    # compare 3 weibos with each other
    #sim1 = model.n_similarity(weibo1, weibo2)
    #sim2 = model.n_similarity(weibo1, weibo3)
    #sim3 = model.n_similarity(weibo2, weibo3)
    #print("similarity between weibo 1 and weibo 2", weibo1, weibo2, sim1)
    #print("similarity between weibo 1 and weibo 3", weibo1, weibo3, sim2)
    #print("similarity between weibo 2 and weibo 3", weibo2, weibo3, sim3)

    # Not all words in each post are in the model.  So, when we try to compare posts we get error messages
    # that the key/word we're trying to compare doesnt exist.
    # To fix this, Write a function that goes through each post and stores the words in the model into a new post
    # So, in the end we will have another set of posts that contain words in the model.
    # Then, we will be able to measure the similarity between posts

    ##########################   Choose a weibo to find similar weibos ###############################
    #weibo1 = final_weibos[3]
    #weibo_input = input("Enter a weibo id between 0 and " + str(max_num_weibos) + " to find similar weibos:")
    weibo_input = st.number_input("Enter a weibo id between 0 and " + str(max_num_weibos) + " to find similar weibos:")
    #print("Finding similar weibos to weibo id:", weibo_input, ".....")
    st.write("Finding similar weibos to weibo id:", weibo_input, ".....")
    #print("\r")
    st.write("\r")

    weibo = int(weibo_input)
    weibo_test = final_weibos[weibo]

    sim_weibos = {}
    sim_list = []  # these are the values for the dict sim_weibos
    weibo_id = []  # these are the keys for the dict sim_weibos
               
    for i in range(len(final_weibos)):
        sim = model.n_similarity(weibo_test, final_weibos[i])
        sim_list.append(sim)
        weibo_id.append(i)

    #print("")
    #print("")
    #print(weibo_id, sim_list)
    #print("")

    for i in weibo_id:
            sim_weibos[i] = sim_list[i]

    #print(sim_weibos)
    #print("")
    sim_weibos_sorted_ascending = dict(sorted(sim_weibos.items(), key=lambda item: item[1])) 
    sim_weibos_sorted_descending = dict(sorted(sim_weibos.items(), key=lambda item: item[1], reverse=True)) 

    #print(sim_weibos_sorted_ascending)
    #print("\r")
    ##print(sim_weibos_sorted_descending)  #Uncomment this if you want to see the similar posts before the weibo_test is removed

    #print("What weibos are similar to weibo id " +str(weibo_input) +" :")
    st.write("What weibos are similar to weibo id " +str(weibo_input) +" :")
    #print(weibo_test)
    st.write(weibo_test)

    # remove the first key/value from the dict since it is comparing the weibo to itself
    # since we sort by ascending order the first key will always be the same as the weibo we're looking to compare
    # we use first_key when we dont know the key.  Comment the line below if you know the key such as
    # when you request input from the user

    #first_key = next(iter(sim_weibos_sorted_descending)) # gets first key in the dict  Use this when we dont know the the key like if we hardcode the weibo to check

    # If you request input from the user then we know the key/weibo id that we should remove from the
    # comparison.  If we know the key use the code line below for same key

    same_key = weibo  # use this when we request input from the user to select the weibo id

    #sim_weibos_sorted_descending.pop(first_key) # removes the first key when we dont know the key, like if we hardcode the key weibo1 for testing
    sim_weibos_sorted_descending.pop(same_key)
    print("\r")
    print(sim_weibos_sorted_descending)
    print("\r")
   

    #print("Top 3 Most Similar Weibos to Weibo ID " +str(weibo_input) + ":")
    st.write("Top 3 Most Similar Weibos to Weibo ID " +str(weibo_input) + ":")
    # Get the keys from sim_weibos_sorted_descending to display the top 3 similar posts
    sim_weibo_ids = list(sim_weibos_sorted_descending.keys())  # list of similar weibo ids compared to the weibo_test
    print("\r")
    st.write("\r")
    for i in range(len(sim_weibo_ids[0:3])):  # sim_weibo_ids[0:4] we only select the top 3 similar weibos
        print("Weibo ID",sim_weibo_ids[i],":",final_weibos[i])
    

    # In[215]:


    #print(final_weibos[1])
    st.write(final_weibos[1])


    # In[423]:


    #################################################################################################### 
    #            Compute the cosine similarity between the first 10 posts and store in sim_matrix
    ####################################################################################################
    st.subheader ("Sim Matrix showing the similarity between the first 10 Weibos")
    temp_list = []
    num_weibos = 10
    sim_matrix = [0]*num_weibos
    #print(sim_matrix)

    for i in range(num_weibos):
        for j in range(num_weibos):
            sim_i_j = model.n_similarity(final_weibos[i], final_weibos[j])
            temp_list.append(sim_i_j)
        sim_matrix[i] = list(temp_list)
        temp_list = []

    #print(len(sim_matrix))
    #print(sim_matrix)
    #print(max(sim_matrix))
    st.table(sim_matrix)
    st.table(max(sim_matrix))
    # In[424]:


    # Program to plot 2-D Heat map
    # using seaborn.heatmap() method
    import numpy as np
    import seaborn as sns
    import matplotlib.pylab as plt
    import streamlit as st

    fig = plt.figure(figsize = (15, 10))
    print(type(sim_matrix))
    print(len(sim_matrix))
    data_set = sim_matrix

    data_set = sim_matrix
    ax = sns.heatmap( data_set , linewidth = 0.2 , cmap = 'coolwarm' )
      
    plt.title( "2-D Heat Map" )
    plt.show()
    st.pyplot(fig)

    # In[426]:


    #sim_input = input("Enter a weibo id between 0 and " + str(num_weibos-1) + " to find the most similar weibo: ")
    sim_input = st.number_input("Enter a weibo id between 0 and " + str(num_weibos-1) + " to find the most similar weibo: ")
    #sim_input = int(sim_input)
    sim_input = st.number_input(sim_input)
    remove_input = sim_matrix[sim_input].pop(sim_input)  # remove the weibo you're searching from the row since it will always be the largest
    #print(remove_input)

    sim_row = sim_matrix[sim_input] # get the row of the weibo

    val, idx = max((val, idx) for (idx, val) in enumerate(sim_row))
    #print(val,idx)
    st.write(val)
    st.write(idx)
    #most_sim = max(sim_matrix[sim_input])
    #print(most_sim)

    # convert list of floats to string
    #result = ", ".join([str(i) for i in sim_matrix[sim_input]])
    #print(result)

    #most_sim_idx = result.index(str(most_sim))
    #print("The most similar weibo is " +str(most_sim)+ " at index", most_sim_idx)

    #print(sorted(sim_matrix[sim_input].index, reverse=True))
    #print(sorted(sim_matrix[sim_input], reverse=True))

