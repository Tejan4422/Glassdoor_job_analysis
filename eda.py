import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('salary_data_cleaned.csv')

def title_simplifier(title):
    if 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'mle'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'na'
    
def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'sr' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
        return 'senior'
    elif 'jr' in title.lower() or 'junior' in title.lower() or 'jr.' in title.lower():
        return 'jr'
    else:
        return 'na'
    
df['job_simp'] = df['Job Title'].apply(title_simplifier)
df.job_simp.value_counts()

df['seniority'] = df['Job Title'].apply(seniority)    
df.seniority.value_counts() 

#fix Los angeles
df.job_state.value_counts()
df['job_state'] = df.job_state.apply(lambda x: x.strip() if x.strip().lower() != 'los angeles' else 'CA') 
df.job_state.value_counts()    

df.drop('job state', inplace = True, axis = 1)
#Competitors count
df['desc_length'] = df['Job Description'].apply(lambda x: len(x))
df['Competitors'] = df['Competitors'].apply(lambda x: len(x.split(',')) if x != '-1' else 0)

#hourly wage to annual
df['min_salary'] = df.apply(lambda x: x.min_salary*2 if x.hourly ==1 else x.min_salary, axis =1 )
df['max_salary'] = df.apply(lambda x: x.max_salary*2 if x.hourly ==1 else x.max_salary, axis =1 )

df['company_txt'] = df.company_txt.apply(lambda x: x.replace('\n', ''))
#df.to_csv('Salary_data_cleaned.csv')
df.describe()