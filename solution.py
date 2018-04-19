import pandas as pd
import numpy as np # linear algebra


from sklearn.model_selection import train_test_split

df = pd.read_csv('train1.csv', index_col=0)
test = pd.read_csv("test (1).csv", index_col=0)



df['birth_year'] = df['birth_year'].str.extract('(\d+)', expand=False)
df['birth_year'] = df['birth_year'].fillna(0)

#test['birth_year'] = test['birth_year'].str.extract('(\d+)', expand=False)
test['birth_year'] = test['birth_year'].fillna(0)

#fillna(s.mean())
df['latitude'] = df['latitude'].fillna(df['latitude'].mean())

test['latitude'] = test['latitude'].fillna(test['latitude'].mean())
df['longitude'] = df['longitude'].fillna(df['longitude'].mean())

test['longitude'] = test['longitude'].fillna(test['longitude'].mean())


df['birth_year'] = df['birth_year'].apply(pd.to_numeric, errors='coerce')


test['birth_year'] = test['birth_year'].apply(pd.to_numeric, errors='coerce')

df['birth_year'] = df['birth_year'].apply(lambda x: 2018-x)

test['birth_year'] = test['birth_year'].apply(lambda x: 2018-x)


#df['birth_year'] = df['birth_year'].apply(lambda x: x - x if x > 100 else x )

#df['birth_year'] = df['birth_year'].apply(lambda x: x - x if x > 100 else x )

#Score : 0.964977257306 vs 0.963369330938

#page_views        0.070592

birthcheck = df.corr()

#print(birthcheck['historical_popularity_index'].sort_values(ascending=False))





# creating dummy variables for the columns that were objects


dummy_research  = pd.get_dummies(df[['sex','country','continent','occupation','industry','domain']])
clean = df.drop(df.columns[[0, 4, 5]], axis=1)
clean_dummy = pd.concat([clean, dummy_research], axis=1)

corr_matrix=clean_dummy.corr()

#print(corr_matrix['historical_popularity_index'].sort_values(ascending=False))


#df['birth_year'] = df['birth_year'].apply(lambda x: x if x<101 else x-x)


#print(data_dummies)


# ADDING DOMAIN INSTITUIONS AND OCCUPATION POLITICIAN

data_dummies = pd.get_dummies(df[['domain']])
data_dummiest = pd.get_dummies(test[['domain']])

pan = data_dummies.drop(data_dummies.columns[[0, 1, 2, 5, 6, 7]], axis=1)
df = pd.concat([pan, df], axis=1)


pant = data_dummiest.drop(data_dummiest.columns[[0, 1, 2, 5, 6, 7]], axis=1)
test = pd.concat([pant, test], axis=1)


# ADDING INDUSTRY-GOvernment AND  OCcupation Politician


data_dummies = pd.get_dummies(df[['sex']])
data_dummiest = pd.get_dummies(test[['sex']])

df = pd.concat([df, data_dummies['sex_Male']], axis=1)
test = pd.concat([data_dummiest['sex_Male'], test], axis=1)


data_dummies = pd.get_dummies(df[['country']])
data_dummiest = pd.get_dummies(test[['country']])

df = pd.concat([df, data_dummies['country_Unknown']], axis=1)
test = pd.concat([data_dummiest['country_Unknown'], test], axis=1)




data_dummies = pd.get_dummies(df[['industry']])
data_dummiest = pd.get_dummies(test[['industry']])

df = pd.concat([df, data_dummies['industry_Government']], axis=1)
test = pd.concat([data_dummiest['industry_Government'], test], axis=1)



data_dummies = pd.get_dummies(df[['industry']])
data_dummiest = pd.get_dummies(test[['industry']])

df = pd.concat([df, data_dummies['industry_Government']], axis=1)
test = pd.concat([data_dummiest['industry_Government'], test], axis=1)





y = df['historical_popularity_index']
X = df[[ 'article_languages', 'birth_year', 'domain_Humanities','domain_Institutions' , 'page_views','average_views','industry_Government', 'sex_Male' , 'latitude','longitude','country_Unknown']]
X = X.applymap(np.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y)



# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 42)




from sklearn.ensemble import RandomForestRegressor



seed = '100'


model = RandomForestRegressor(n_estimators=200, min_samples_split=5, min_samples_leaf=1,
                              max_features="auto", max_depth=10, bootstrap=True, random_state=int(seed), )     #Hypertuned

model.fit(train_features,train_labels)
print("Score : %s" %model.score(train_features,train_labels))


#test['historical_popularity_index'] = model.predict(test[[ 'article_languages', 'birth_year', 'domain_Humanities','domain_Institutions', 'page_views','average_views', ]].values)

test['historical_popularity_index']  = model.predict(test[[ 'article_languages', 'birth_year', 'domain_Humanities','domain_Institutions', 'page_views','average_views',
                                                                 'industry_Government', 'sex_Male', 'latitude','longitude','country_Unknown']].values)

test[['historical_popularity_index']].to_csv('sample_submission.csv')

