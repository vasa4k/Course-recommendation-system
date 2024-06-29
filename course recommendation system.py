#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install neattext')


# In[2]:


get_ipython().system('pip install scikit-learn')


# In[3]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[4]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import neattext.functions as nfx


# In[5]:


data = pd.read_csv(r"D:\admin\Downloads\udemy_courses.csv")


# In[6]:


data.head()


# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# In[9]:


data.describe()


# In[10]:


data.duplicated().any()


# In[11]:


data[data.duplicated()]


# In[12]:


data = data.drop_duplicates()


# In[13]:


data.shape


# In[14]:


data['course_title']


# In[15]:


data.columns


# ### Popularity-Based recommendation system

# In[16]:


def pbs(df,top_n=5):
    data['popularity_score'] = 0.6 * data['num_subscribers'] + 0.4 * data['num_reviews']
    df_sorted = data.sort_values(by='popularity_score',ascending = False)
    recommended_courses = df_sorted[['course_title','popularity_score']].head(top_n)
    return recommended_courses


# In[17]:


pbs(data)


# ### Content-Based Recommendation System 

# In[22]:


data['course_title'] = data['course_title'].apply(nfx.remove_stopwords)
data['course_title'] = data['course_title'].apply(nfx.remove_special_characters)


# In[25]:


data.sample(5)


# In[26]:


data['title_subject'] = data['course_title'] +' '+data['subject']


# In[27]:


cv = CountVectorizer(max_features = 3000)
vectors = cv.fit_transform(data['title_subject']).toarray()


# In[28]:


vectors[0]


# In[30]:


len(cv.get_feature_names_out())


# In[31]:


similarity = cosine_similarity(vectors)


# In[32]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[33]:


def recommend(course):
    # let's featch the index
    course_index = data[data['course_title']==course].index[0]
    distances = similarity[course_index]
    courses_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in courses_list:
        print(data.iloc[i[0]]['course_title'])


# #### RESULTS 

# In[34]:


recommend("know HTML Learn HTML Basics")


# In[36]:


recommend('Complete Investment Banking Course 2017')


# In[42]:


recommend('Essentials money value financial Life')


# In[55]:


recommend("Projects Django Python")


# In[58]:


recommend("Introduction Data Analytics Microsoft Power BI")


# In[61]:


recommend('Introduction Cryptocurrencies Blockchain')

