import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Salary_data_cleaned.csv')

df.columns

df.age.hist()

df.desc_length.hist()
df.Rating.hist()
df.boxplot(column = 'Rating')

corr = df[['age', 'avg_salary', 'Rating', 'desc_length', 'Competitors']].corr()
cmap = sns.diverging_palette(220, 10, as_cmap = True)
sns.heatmap(corr, vmax = .3, center = 0, square = True, linewidths = 5, cbar_kws = {"shrink": 0.5}, cmap = cmap)

df_cat = df[['Location', 'Headquarters', 'Size','Type of ownership', 'Industry', 'Sector', 'Revenue',
             'company_txt', 'job_state','python', 'R', 'aws',
       'spark', 'excel', 'job_simp', 'seniority',]]

for i in df_cat.columns:
    cat_num = df_cat[i].value_counts()[:20]
    print('grapth for %s total %d'%(i, len(cat_num)))
    chart = sns.barplot(x = cat_num.index, y = cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation = 90)
    plt.show()
    
pd.pivot_table(df, index = 'job_simp', values = 'avg_salary')

pd.pivot_table(df, index = ['job_simp', 'seniority'], values = 'avg_salary')
pd.pivot_table(df, index = 'job_state', values = 'avg_salary').sort_values('avg_salary', ascending = False)

pd.pivot_table(df[df.job_simp == 'data scientist'], index = 'job_state', values = 'avg_salary').sort_values('avg_salary', ascending = False)

df_pivots = df[['Rating', 'Industry', 'Sector', 'Revenue', 'Competitors','hourly', 'employer_provided','python', 'R', 'aws',
       'spark', 'excel','Type of ownership', 'avg_salary']]

for i in df_pivots.columns:
    print(i)
    print(pd.pivot_table(df_pivots, index = i,values = 'avg_salary').sort_values('avg_salary',ascending=False))

pd.pivot_table(df_pivots, index = 'Revenue', columns = 'python', values = 'avg_salary', aggfunc = 'count')
#word map 
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

words = " ".join(df['Job Description'])
def punctuation_stop(text):
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered
    
words_filtered = punctuation_stop(words)
text = " ".join([ele for ele in words_filtered])

wc = WordCloud(background_color = 'white', random_state = 1, stopwords = STOPWORDS, max_words = 2000,
               width = 800, height = 1500)
wc.generate(text)
plt.figure(figsize = [10,10])
plt.imshow(wc, interpolation = "bilinear")
plt.axis('off')
plt.show()