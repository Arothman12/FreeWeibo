import pandas as pd
import re
import streamlit as st

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
print(final_weibos_new[2])

##print(len(final_weibos))
print(len(final_weibos_new))


# In[63]:


post_list_sample_new = post_list_sample_new.values.tolist()


# In[64]:


# Applying the model.  

# The last step would be to use the model to do something. As it turns out Gensim lets you do 
# a lot of things with your model. Below are some examples Python

import gensim

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
#weibo_input = input("Enter a weibo id between 0 and " + str(max_num_weibos) + " to find similar weibos:")
label = "Enter a weibo id between 0 and " + str(max_num_weibos - 1) + " to find similar weibos:"
weibo_input = st.number_input(label, 0, max_num_weibos-1, value=0, step=1)
#print("Finding similar weibos to weibo id:", weibo_input, ".....")

st.write("These are similar weibos to weibo id entered:", weibo_input, ".....")
#print("\r")
st.write("\r")

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

#print("What weibos are similar to weibo id " +str(weibo_input) +" :")
st.write("What weibos are similar to weibo id" +str(weibo_input) +" :")
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

print("Top 3 Most Similar Weibos to Weibo ID " +str(weibo_input) + ":")
st.write("Top 3 Most Similar Weibos to Weibo ID " +str(weibo_input) + ":")
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

#for i in range(len(final_weibos_new)):
#    final_weibos_new[i] = list(filter(('""""').__ne__, final_weibos_new[i]))

#print(final_weibos[110])
#print(final_weibos_new[110])

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
        sim_i_j = model.n_similarity(final_weibos[i], final_weibos[j])
        temp_list.append(sim_i_j)
    sim_matrix[i] = list(temp_list)
    temp_list = []

#print(len(sim_matrix))
#print(sim_matrix)
st.table(sim_matrix)
#print(max(sim_matrix))
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


sim_input = input("Enter a weibo id between 0 and " + str(num_weibos-1) + " to find the most similar weibo: ")
sim_input = int(sim_input)
remove_input = sim_matrix[sim_input].pop(sim_input)  # remove the weibo you're searching from the row since it will always be the largest
#print(remove_input)

sim_row = sim_matrix[sim_input] # get the row of the weibo

val, idx = max((val, idx) for (idx, val) in enumerate(sim_row))
print(val,idx)

#most_sim = max(sim_matrix[sim_input])
#print(most_sim)

# convert list of floats to string
#result = ", ".join([str(i) for i in sim_matrix[sim_input]])
#print(result)

#most_sim_idx = result.index(str(most_sim))
#print("The most similar weibo is " +str(most_sim)+ " at index", most_sim_idx)

#print(sorted(sim_matrix[sim_input].index, reverse=True))
#print(sorted(sim_matrix[sim_input], reverse=True))

