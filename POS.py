#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from avidanwordembeddings import WEBIO
title = st.title('Welcome to my FreeWeibo Dataset')
#st.write(title)
# In[15]:

def POSMain():

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
    new_df.head()

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
    word_df.head()


    # In[22]:


    #Get the part of speech (pos) for each word in the segnmented_content_list

    import jieba
    import jieba.posseg as pseg
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
    #        print('%s %s' % (word, flag))

    pos_df = pd.DataFrame({'post_num': row_num, 'word': post, 'pos': pos})
    pos_df.head()


    # In[23]:


    # print the number of pos tags for each post

    pos_df.head()
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
    df_top_pos = pos_df.groupby('pos')['pos'].count().reset_index(name='count').sort_values(['count'],ascending=False).head(15)
    df_top_pos



    Count_Chart_By_POS = (
        Bar()
        .add_xaxis(df_top_pos['pos'].to_list())
        .add_yaxis("",df_top_pos['count'].to_list())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Count of Total Number of Postions Based on Users Posts", subtitle = "2014-2022 Data")
            )  
    )
    st_pyecharts(Count_Chart_By_POS)



POS = st.button("Parts of Speech", on_click = POSMain,args=None)
WEB = st.button("Word Embeddings", on_click = WEBIO,args=None)
