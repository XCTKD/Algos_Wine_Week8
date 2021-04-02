#!/usr/bin/env python
# coding: utf-8

# # Machine learning on wine
# 
# **Topics:** Text analysis, linear regression, logistic regression, text analysis, classification
# 
# **Datasets**
# 
# - **wine-reviews.csv** Wine reviews scraped from https://www.winemag.com/
# - **Data dictionary:** just go [here](https://www.winemag.com/buying-guide/tenuta-dellornellaia-2007-masseto-merlot-toscana/) and look at the page
# 
# ## The background
# 
# You work in the **worst newsroom in the world**, and you've had a hard few weeks at work - a couple stories killed, a few scoops stolen out from under you. It's not going well.
# 
# And because things just can't get any worse: your boss shows up, carrying a huge binder. She slams it down on your desk.
# 
# "You know some machine learning stuff, right?"
# 
# You say "no," but she isn't listening. She's giving you an assignment, the _worst assignment_...
# 
# > Machine learning is the new maps. Let's get some hits!
# >
# > **Do some machine learning on this stuff.**
# 
# "This stuff" is wine reviews.
# 
# ## A tiny, meagre bit of help
# 
# You have a dataset. It has some stuff in it:
# 
# * **Numbers:**
#     - Year published
#     - Alcohol percentage
#     - Price
#     - Score
#     - Bottle size
# * **Categories:**
#     - Red vs white
#     - Different countries
#     - Importer
#     - Designation
#     - Taster
#     - Variety
#     - Winery
# * **Free text:**
#     - Wine description

# # Cleaning up your data
# 
# Many of these pieces - the alcohol, the year produced, the bottle size, the country the wine is from - aren't in a format you can use. Convert the ones to numbers that are numbers, and extract the others from the appropriate strings.

# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv("wine-reviews.csv")


# In[6]:


df.columns


# In[7]:


df.head()


# ## What might be interesting in this dataset?
# 
# Maybe start out playing around _without_ machine learning. Here are some thoughts to get you started:
# 
# * I've heard that since the 90's wine has gone through [Parkerization](https://www.estatewinebrokers.com/blog/the-parkerization-of-wine-in-the-1990s-and-beyond/), an increase in production of high-alcohol, fruity red wines thanks to the influence of wine critic Robert Parker.
# * Red and white wines taste different, obviously, but people always use [goofy words to describe them](https://winefolly.com/tutorial/40-wine-descriptions/)
# * Once upon a time in 1976 [California wines proved themselves against France](https://en.wikipedia.org/wiki/Judgment_of_Paris_(wine)) and France got very angry about it

# In[ ]:





# In[ ]:





# In[ ]:





# ## But machine learning?
# 
# Well, you can usually break machine learning down into a few different things. These aren't necessarily perfect ways of categorizing things, but eh, close enough.
# 
# * **Predicting a number**
#     - Linear regression
#     - For example, how does a change in unemployment translate into a change in life expectancy?
# * **Predicting a category** (aka classification)
#     - Lots of algos options: logistic regression, random forest, etc
#     - For example, predicting cuisines based on ingredients
# * **Seeing what influences a numeric outcome**
#     - Linear regression since the output is a number
#     - For example, minority and poverty status on test scores 
# * **Seeing what influences a categorical outcome**
#     - Logistic regression since the output is a category
#     - Race and car speed for if you get a waring vs ticket
#     - Wet/dry pavement and car weight if you survive or not in a car crash)
# 
# We have numbers, we have categories, we have all sorts of stuff. **What are some ways we can mash them together and use machine learning?**
# 
# ### Brainstorm some ideas
# 
# Use the categories above to try to come up with some ideas. Be sure to scroll up where I break down categories vs numbers vs text!
# 
# **I'll give you one idea for free:** if you don't have any ideas, start off by creating a classifier that determines whether a wine is white or red based on the wine's description.

# Predicting a wine's score based on how much it cost. (Linear Regression)

# Predicting the alcoholic content based on the date published. (Linear Regression)

# In[ ]:





# In[ ]:





# You can also go to https://library.columbia.edu and see if you can find some academic papers about wine. I'm sure they'll inspire you! (and they might even have some ML ideas in them you can steal, too)

# # Implement 2 of your machine learning ideas

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




