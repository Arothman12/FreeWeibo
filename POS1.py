#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from streamlit_option_menu import option_menu
with st.sidebar:
    selected = option_menu("Free Weibo Data Analysis",["Main Page","Parts of Speech","Word Embedding","LDA Topic Analysis"],icons=['display','translate','box-arrow-in-right','fingerprint'],default_index=1)


# In[15]:
while selected == "Main Page":
    st.title("Analytic Techniques to Visualize Censored Content (Sina Weibo)")
    st.subheader("Designed by Avidan Rothman (Advised by Dr. Leberknight)")
    st.write("https://freeweibo.com/")
    st.caption("FreeWeibo.com, an anonymous and unblocked Sina Weibo search, was established on October 11, 2012 by GreatFire.org. Free Weibo captures and saves some Weibo posts that have been censored by Sina Weibo or deleted by users. You can search for any keyword, including blocked sensitive words.")
    st.write("https://en.greatfire.org/")
    st.caption("We are an anonymous organization based in China. We launched our first project in 2011 in an effort to help bring transparency to online censorship in China. Now we focus on helping Chinese to freely access information. Apart from being widely discussed in most major mass media, GreatFire has also been the subject of a number of academic papers from various research institutions. FreeWeibo.com won the 2013 Deutsche Welle “Best Of Online Activism” award in the “Best Innovation” category. In 2016, GreatFire won a Digital Activism fellowship from Index on Censorship.")
    st.write("https://startpage.freebrowser.org/zh/")
    st.caption("We have directed Chinese internet users more than 13 million times to censored news stories about government corruption, politics, scandals and other sensitive information.")
    st.image("WeiboCensorship.jpg", caption = "Image from https://cpj.org/2016/03/read-and-delete-how-weibos-censors-tackle-dissent/")
    st.write("Censorship is one of the most parasitic ethical and moral problems the world is combatting today. It has infected every bit of the global social fabric makeup and we are seeing it turn entire countries upside down. China has been one of those countries that has been guilty of this debilitating practice for quite some time, and they have become leaders in the evolution of it as well.")
    break
while selected == "Parts of Speech":
# https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/Multilingual/Chinese/03-POS-Keywords-Chinese.html
    import spacy
    from spacy import displacy
    from collections import Counter
    import pandas as pd
    #pd.set_option("max_rows", 400)
    #pd.set_option("max_colwidth", 400)

    # Download Language Model
    #!python -m spacy download zh_core_web_md


    # In[16]:


    # Load Language Model
    nlp = spacy.load('zh_core_web_md')


    # In[17]:


    # Read in the data

    import pandas as pd
    import re
    import streamlit as st
    
    df = pd.read_excel("C:\\Users\\avida\\freeweibo-09-15-2022.xlsx")
    df.index.names = ['row_id']

    new_df = df[['User_name', 'time_created', 'OriginalPostLink', 'HotTerm', 'content']]
    new_df = new_df.dropna() # If column value is empty drop the entire row

    weibo_id_list = []

    for ind in new_df.index:
        row_content = new_df['OriginalPostLink'][ind]
        weibo_id = re.findall(r'\d+', str(row_content)) or ["Error"]
    #    print(weibo_id[0])
        weibo_id_list.append(weibo_id[0])
        
    #print(weibo_id_list)
    new_df['weibo_ids'] = weibo_id_list

    # Dropping rows that contain ['0'] as a weibo id
    ###new_df = new_df[new_df["weibo_ids"].str.contains("0") == False]

    # new_df.head()
    print("Number of rows in original data: ", len(df))
    print("Number of rows after removing empty rows in columns:", len(new_df))
    print("Number rows removed:", len(df) - len(new_df))
    st.write("Number of rows in original data: ", len(df))
    st.write("Number of rows after removing empty rows in columns: ", len(new_df))
    st.write("Number rows removed:", len(df) - len(new_df))
    st.header("Free Weibo Dataset")
    #new_df.head()
    # In[19]:


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
    special_char = re.compile('[《【】VS.！→→？～。：，//,@_!#$%^&*()<>?/\|}{~:]')
    #new_df['symbols'] = new_df['content'].str.extract(special_char, expand=False)
    new_df['content'] = new_df['content'].str.replace(special_char,'')

    #new_df['content'] = new_df['content'].str.replace('http://', '')        # remove http:// from each row
    #new_df['content'] = new_df['content'].apply(lambda x: re.split('http:\/\/.*', str(x))[0])
    #new_df['content'] = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
                               
    filter1_df = new_df[new_df["content"].str.contains(r'[\u4e00-\u9FFF]')==True]  # Keep only Chinese characters

    filter1_df


    # In[20]:


    # Segment the cleaned content.  This will be necessary to do any pos tagging

    import pandas as pd
    import re
    import jieba
    import pinyin
    import unicodedata
    import streamlit as st 
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

    # In[21]:


    word_df = pd.DataFrame({'post_num': row_num, 'post': segmented_content_list})
    #word_df.head()


    # In[22]:


    #Get the part of speech (pos) for each word in the segnmented_content_list

    import jieba
    import jieba.posseg as pseg
    import streamlit as st
    st.header("Getting the part of speech for each word found in the segmented dataset.")
    #words = pseg.cut("我爱北京天安门") #jieba默认模式
    #jieba.enable_paddle() #启动paddle模式。 0.40版之后开始支持，早期版本不支持
    #words = pseg.cut("我爱北京天安门",use_paddle=True) #paddle模式

    post = []
    pos = []
    row_num = []
        
    for i in filter1_df.index:
        post_pos = filter1_df['content'][i]
        words = pseg.cut(post_pos)
        for word, flag in words:
            row_num.append(i)       
            post.append(word)
            pos.append(flag)
    pos_df = pd.DataFrame({'post_num': row_num, 'word': post, 'pos': pos})
    #pos_df.head()
    st.write(pos_df)

    # In[23]:


    # print the number of pos tags for each post
    st.header("Number of pos tags for each post")
    #pos_df.head()
    #print(len(pos_df))
    pos_df.dtypes
    n_by_post = pos_df.groupby("post_num")["pos"].count()
    n_by_post.head(10)
    #pos_df.post_num.unique()


    # In[24]:


    # https://stackoverflow.com/questions/22219004/how-to-group-dataframe-rows-into-list-in-pandas-groupby

    pos_df.groupby('post_num')['pos'].apply(list)
    pos_df1 = pos_df.groupby('post_num')['pos'].apply(list).reset_index(name='pos_list')
    pos_df1


    # In[25]:


    # MERGE the POS, WORD and FILTER1 dataframe into the FILTER1 datafram

    filter1_df['words'] = word_df['post']
    filter1_df['pos'] = pos_df1['pos_list']
    final_df = filter1_df
    final_df.head()
    print(final_df['content'][0])
    print(final_df['words'][0])
    print(final_df['words'][0][0], final_df['pos'][0][0])


    # In[26]:


    import matplotlib
    # https://www.dataknowsall.com/pos.html
    # count total # of pos for all posts

    #pos_df.head()
    from streamlit_echarts import st_echarts
    from pyecharts import options as opts
    from pyecharts.charts import Bar
    from streamlit_echarts import st_pyecharts
    import streamlit as st
    st.header("Count total of POS for all posts")
    df_top_pos = pos_df.groupby('pos')['pos'].count().reset_index(name='count').sort_values(['count'],ascending=False).head(15)
    df_top_pos
    st.write(df_top_pos)

    Count_Chart_By_POS = (
        Bar()
        .add_xaxis(df_top_pos['pos'].to_list())
        .add_yaxis("",df_top_pos['count'].to_list())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Count of Total Number of Postions Based on Users Posts", subtitle = "2014-2022 Data")
            )  
    )
    st_pyecharts(Count_Chart_By_POS)

    # Get all nouns in the data
    # https://www.dataknowsall.com/pos.html
    #pos_df.head()
    import streamlit as st
    st.header("Getting all the nouns in dataset")
    df_nn = pos_df[pos_df['pos'] == 'n'].copy()
    df_nn.groupby('word')['word'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(10)
    st.write(df_nn.groupby('word')['word'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(10))
    

    import matplotlib.pyplot as plt
    from itertools import cycle
    from streamlit_echarts import st_echarts
    from pyecharts import options as opts
    from pyecharts.charts import Bar
    from streamlit_echarts import st_pyecharts
    plt.rcParams['font.sans-serif'] = ['simsun'] 
    plt.rcParams['font.family'] ='sans-serif'

    # creating the bar plot
    #https://www.dataknowsall.com/pos.html
    df_top_nn = df_nn.groupby('word')['word'].count().reset_index(name='count').\
        sort_values(['count'], ascending=False).head(10)
    df_top_nn

    Ten_Nouns_By_Freq = (
        Bar()
        .add_xaxis(df_top_nn['word'].to_list())
        .add_yaxis("",df_top_nn['count'].to_list())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Top 10 most frequently used nouns")
            )  
    )
    st_pyecharts(Ten_Nouns_By_Freq)
    break

while selected == "Word Embedding":
    import streamlit as st
    import pandas as pd
    import re

    
    post_list_sample_new = pd.read_csv("C:\\Users\\avida\\FreeWeibo\\post_list_sample.csv")
    post_list_sample_new.head()
    post_list_sample_new.values.tolist()


    # In[62]:


    # Read in the the final_weibos.csv
    # https://stackoverflow.com/questions/16923281/writing-a-pandas-dataframe-to-csv-file

    import pandas as pd
    import re

    #print(final_weibos[0:3])
    final_weibos_new = pd.read_csv("C:\\Users\\avida\\FreeWeibo\\final_weibos.csv", sep = '/t' ,header=None, engine = 'python')
    #final_weibos_new.head()
    final_weibos = final_weibos_new
    final_weibos_new = final_weibos_new.values.tolist()


    #x = final_weibos_new[1]
    #newlist = [word for line in x for word in line.split(",")]
    #print(newlist)
    new_list =[]

    for i in range(len(final_weibos_new)):
        weibo_post = final_weibos_new[i]
        weibo_post_words = [word for line in weibo_post for word in line.split(",")]
        new_list.append(weibo_post_words)

    ##print(final_weibos[2])

    ##print(new_list[2])
    final_weibos_new = new_list
    print(final_weibos_new[10])

    ##print(len(final_weibos))
    print(len(final_weibos_new))


    # In[63]:


    post_list_sample_new = post_list_sample_new.values.tolist()


    # In[64]:


    # Applying the model.  

    # The last step would be to use the model to do something. As it turns out Gensim lets you do 
    # a lot of things with your model. Below are some examples Python

    import gensim
    import streamlit as st
    st.header("Applying the model")
    model = gensim.models.KeyedVectors.load_word2vec_format('word2vec.vector', binary=False)

    #How many words does our model have? We simply need to check the size of our vocabulary with the Python len function:
    #print(len(model))

    # What is the upper bound of the number of weibos? We use this upper bound number in our input statement on line 51
    max_num_weibos = len(post_list_sample_new)

    #print the word vector associated with the word
    #print(model["中国"])
    #print(model["人"])  # Most frequently used noun in the Freeweibo dataset

    # print top 10 most similar words to the input word
    #print(model.most_similar(positive='广东', negative=None, topn=10, restrict_vocab=None, indexer=None))
    ##print(model.most_similar(positive='人', negative=None, topn=10, restrict_vocab=None, indexer=None))

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
    
    label = "Enter a weibo id between 0 and " + str(max_num_weibos - 1) + " to find similar weibos:"
    weibo_input = st.number_input(label, 0, max_num_weibos-1, value=0, step=1, key="Initial")
    if not weibo_input:
        st.stop()
    #weibo_input = input("Enter a weibo id between 0 and " + str(max_num_weibos) + " to find similar weibos:")
    st.write("Finding similar weibos to weibo id:", weibo_input, ".....")
    #print("Finding similar weibos to weibo id:", weibo_input, ".....")
    print("\r")

    weibo= int(weibo_input)
    weibo_test = final_weibos_new[weibo]

    sim_weibos = {}
    sim_list = []  # these are the values for the dict sim_weibos
    weibo_id = []  # these are the keys for the dict sim_weibos
               
    for i in range(len(final_weibos_new)):
        sim = model.n_similarity(weibo_test, final_weibos_new[i])
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
    st.write("What weibos are similar to weibo id " +str(weibo_input) +" :")
    #print("What weibos are similar to weibo id " +str(weibo_input) +" :")

    #print(weibo_test)
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
    st.write("Top 3 Most Similar Weibos to Weibo ID " +str(weibo_input) + ":")
    #print("Top 3 Most Similar Weibos to Weibo ID " +str(weibo_input) + ":")
    # Get the keys from sim_weibos_sorted_descending to display the top 3 similar posts
    sim_weibo_ids = list(sim_weibos_sorted_descending.keys())  # list of similar weibo ids compared to the weibo_test
    print("\r")
    for i in range(len(sim_weibo_ids[0:3])):  # sim_weibo_ids[0:4] we only select the top 3 similar weibos
        #print("Weibo ID",sim_weibo_ids[i],":",final_weibos_new[i])
        st.write("Weibo ID",sim_weibo_ids[i],":",final_weibos_new[i])

    # In[56]:


    # WE DONT NEED THIS CODE.  IT IS ONLY USED TO FIND BAD CHARACTERS AND VERIFY THAT THEY WERE REMOVED IN PREVIOUS STEPS

    #print(final_weibos_new[110])
    #print(final_weibos[110])
    #word = '27'
    #word = '""""'
    #for i in range(len(final_weibos_new)):
    #    if word in final_weibos_new[i]:
    #        print(i,word)

    for i in range(len(final_weibos_new)):
        final_weibos_new[i] = list(filter(('""""').__ne__, final_weibos_new[i]))

    #print(final_weibos[110])
    print(final_weibos_new[110])


    # In[423]:


    #################################################################################################### 
    #            Compute the cosine similarity between the first 10 posts and store in sim_matrix
    ####################################################################################################

    temp_list = []
    num_weibos = 10
    sim_matrix = [0]*num_weibos
    #print(sim_matrix)

    for i in range(num_weibos):
        for j in range(num_weibos):
            sim_i_j = model.n_similarity(final_weibos_new[i], final_weibos_new[j])
            temp_list.append(sim_i_j)
        sim_matrix[i] = list(temp_list)
        temp_list = []

    #print(len(sim_matrix))
    print(sim_matrix)
    print(max(sim_matrix))


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

    label1 = "Enter a weibo id between 0 and " + str(num_weibos-1) + " to find the most similar weibo: "
    weibo_input1 = st.number_input(label1, 0, num_weibos-1, value=0, step=1,key="Similar")
    if not weibo_input1:
        st.stop()
    #sim_input = input("Enter a weibo id between 0 and " + str(num_weibos-1) + " to find the most similar weibo: ")
    sim_input = int(weibo_input1)
    remove_input = sim_matrix[sim_input].pop(sim_input)  # remove the weibo you're searching from the row since it will always be the largest
    #print(remove_input)

    sim_row = sim_matrix[sim_input] # get the row of the weibo

    val, idx = max((val, idx) for (idx, val) in enumerate(sim_row))
    print(val,idx)
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

    break
while selected == "LDA Topic Analysis":
   
    import pandas as pd
    import streamlit as st
    #df = pd.read_excel("C:\\Users\leberknightc\\freeweibo-09-15-2022.xlsx")
    df = pd.read_excel("C:\\Users\\avida\\freeweibo-09-15-2022.xlsx")
    #df.index.names = ['row_id']

    #df = df[df['HotTerm'].notna()] #drops rows if nan appears in the HotTerm column
    #df = df[df['content'].notna()] #drops rows if nan appears in the content column

    #df['content']= df['content'].replace('\n',' ', regex=True).replace('\t',' ', regex=True)


    df.head()  # view the first 5 rows in the data frame
    #row_1=df.iloc[0]  #look at only first row in data frame
    #print(row_1)

    #content = df["content"]  # look at only the content column
    #content.head()


    # In[14]:


    # Remove rows where content is 'NaN'

    # df = df.dropna()  drop rows that contain 'NaN' in any column
    df = df.dropna(subset=['content'])  # drop rows that contain 'NaN' in the content column

    # Hash the content column to check for any duplicate weibo posts

    df['hash'] = df['content'].apply(hash)


    # Selecting duplicate rows except first
    # occurrence based on all columns
    duplicate = df[df.duplicated()]
     
    print("Duplicate Rows :")
     
    # Print the resultant Dataframe
    #duplicate['content']


    # In[15]:


    # Remove the columns we're not interested in 
    #https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0

    new_df= df.drop(columns=['FreeWeibo_Post_Id', 'repostscount', 'censored', 'deleted', 'adult_keyword', 'censored_keyword', 'OriginalPostLink', 'time_scrapped', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21'], axis=1).sample(100)
    # Print out the first rows of data
    new_df.head()

    # In[16]:


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


    # In[17]:


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


    # In[18]:


    # Take the words from each post in the segmented_content_list and merge all the words into 1 long string for our 
    # word cloud

    long_string = ' '
    for i in range(len(segmented_content_list)):
        long_string += ','.join(segmented_content_list[i])
            
    print(long_string)    


    # In[19]:

    # In[17]:


    #                                   Test removing stop words
    # encoding: utf-8
    import codecs
    import jieba
    import io

    stop_words=[]

    if __name__ == "__main__":
        str_in = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造上"
        stopwords = io.open('stopwords-zh.txt', mode = 'r', encoding= 'utf-8')
        for i in stopwords:
            stop_words.append(i)
        chinese_stop_words = ''.join(stop_words)
    #    stopwords = codecs.open('stopwords.zh.txt', 'r', 'utf-8').read().split(',')
        seg_list = jieba.cut_for_search(str_in)
        for seg in seg_list:
            if seg not in chinese_stop_words:
                print(seg)


    # In[20]:


    #                                   Removing stop words from long_string for WordCloud
    # encoding: utf-8
    import codecs
    import jieba
    import io

    stop_words = []
    new_long_string = []

    if __name__ == "__main__":
    #    str_in = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造上"
        stopwords = io.open('stopwords-zh.txt', mode = 'r', encoding= 'utf-8')
        for i in stopwords:
            stop_words.append(i)
        chinese_stop_words = ''.join(stop_words)
    #    stopwords = codecs.open('stopwords.zh.txt', 'r', 'utf-8').read().split(',')
        seg_list = long_string
        for seg in seg_list:
            if seg not in chinese_stop_words:
    #            print(seg)
                new_long_string.append(seg)
        clean_long_string = ','.join(new_long_string)


    # In[21]:



    # Create WordCloud
    # Import the wordcloud library
    import streamlit as st
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    st.header("WordCloud created from words in dataset")
    # Create a WordCloud object
    wordcloud = WordCloud(font_path= "C:\\windows\\Fonts\\simsun", background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

    # Generate a word cloud
    wordcloud.generate(long_string)
    plt.figure(figsize = (15, 10))

    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    #plt.show()
    st.pyplot(plt.gcf())
    # Visualize the word cloud
    #wordcloud.to_image()


    # In[22]:


    # create a dictionary of word frequencies
    text_dictionary = wordcloud.process_text(clean_long_string)
    # sort the dictionary
    word_freq={k: v for k, v in sorted(text_dictionary.items(),reverse=True, key=lambda item: item[1])}

    #use words_ to print relative word frequencies
    rel_freq=wordcloud.words_

    #print results
    print(list(word_freq.items())[:5])
    print(list(rel_freq.items())[:5])


    # In[44]:


    weibo_words = pd.DataFrame({'post_num': row_num, 'words': segmented_content_list})
    weibo_words.head()
    #print(weibo_words['words'][0])


    # In[74]:


    #                                         Test removing multiple commas in a string

    import re
    first_string = "h,ard,,,c,,ode,p,rogr,,, ammer"
    second_string = re.sub(","," ",first_string)
    #second_string = re.sub(" ",",", first_string)
    third_string = re.sub("\s+", ",", second_string.strip())
    print(first_string)
    print(second_string)
    print(third_string)


    # In[53]:


    #                      Remove stop words from corpus for LDAvis

    # Step 1: iterate through each post and each word in post to check if the word is a stop word
    # Step 2: remove stopword


    import gensim.corpora as corpora

    new_segmented_content_list = []
    clean_weibo = []
    clean_segmented_content_list = []

    for i in range(len(segmented_content_list)):
        clean_weibo.append([])
        weibo_content = segmented_content_list[i]
    #    print(weibo_content)
        for word in weibo_content:
            if word not in chinese_stop_words:
                clean_weibo[i].append(word)
    #        print(clean_weibo)
        new_segmented_content_list.append(clean_weibo)
    #    else:
    #        print("Contains stopword")

    for j in range(len(new_segmented_content_list)):
        clean_segmented_content_list.append(new_segmented_content_list[0][j])


    # In[57]:


    #  Just testing the length of posts before and after stop words removed.  Dont need this code for streamlit

    print(len(segmented_content_list[0]))
    #print(new_segmented_content_list[0][0:10])
    print(len(new_segmented_content_list[0][0]))
    print(len(clean_weibo[0]))
    print(len(clean_segmented_content_list[0]))


    # In[58]:


    import gensim.corpora as corpora
    # Create Dictionary
    id2word = corpora.Dictionary(segmented_content_list)

    # Create Corpus
    texts = segmented_content_list

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    print(corpus[:1][0][:30])

    #st.markdown(corpus[:1][0][:30])
    # In[22]:


    # LDA Model Training
    #import gensim
    #from pprint import pprint
    # number of topics
    #num_topics = 10
    # Build LDA model
    #lda_model = gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=num_topics)
    #Print the Keyword in the 10 topics
    #pprint(lda_model.print_topics())
    #doc_lda = lda_model[corpus]
    #st.write(lda_model.print_topics())
    #st.write(lda_model[corpus])
    # In[23]:


    #import pyLDAvis
    #import pyLDAvis.gensim
    #import pyLDAvis.gensim_models

    #import os
    #import pickle 

    # Visualize the topics
    #pyLDAvis.enable_notebook()
    #path = r'C:\\Users\\avida\\FreeWeibo'
    # LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
    #LDAvis_data_filepath = os.path.join(path, 'ldavis_prepared_'+str(num_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    #if 1 == 1:
    #    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    #    with open(LDAvis_data_filepath, 'wb') as f:
    #        pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    #with open(LDAvis_data_filepath, 'rb') as f:
    #    LDAvis_prepared = pickle.load(f)
    #pyLDAvis.save_html(LDAvis_prepared, r'C:\\Users\\avida\\FreeWeibo\\ldavis_prepared'+ str(num_topics) +'.html')
    #pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')
    #LDAvis_prepared
    import streamlit as st
    st.header("LDA Visualization")
    st.image("LDAvis.png",caption="This functionality unfortunately cannot be incorporated into streamlit at this time but below is an image of the visualization")
    break

