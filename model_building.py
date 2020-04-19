import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Salary_data_cleaned.csv')

#choose relevent columns
#choose dummy data
# train test split

#multiple linera, lasso regre, random forest 
#tune with gridsearchcv
df.columns
df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','Competitors', 'hourly','employer_provided',
               'job_state','same_state','age','python','spark','aws','excel','job_simp','seniority','desc_length']]
df_dum = pd.get_dummies(df_model)

from sklearn.model_selection import train_test_split
X = df_dum.drop('avg_salary', axis =1)
y = df_dum.avg_salary.values

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

import statsmodels.api as sm
X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
lm = LinearRegression()
lm.fit(X_train, y_train)
np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv =3))

lm_l = Lasso()
np.mean(cross_val_score(lm_l, X_train, y_train, scoring = 'neg_mean_absolute_error', cv =3))
alpha = []
error = []
for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha = (i)/100)
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring = 'neg_mean_absolute_error', cv =3)))

plt.plot(alpha, error)
err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv =3))

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators' : range(10,300,10), 'criterion' : ('mse','mae'), 'max_features' : ('auto','sqrt','log2')}

gs = GridSearchCV(rf, parameters, scoring = 'neg_mean_absolute_error', cv = 3)
gs.fit(X_train, y_train)