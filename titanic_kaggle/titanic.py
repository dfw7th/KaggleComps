#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


train = pd.read_csv("/home/fw7th/Documents/ML_datasets/Kaggle_comp/titanic/train.csv")
test = pd.read_csv("/home/fw7th/Documents/ML_datasets/Kaggle_comp/titanic/test.csv")


# ## Clean and Transform Stage

# In[3]:


# Ok lets make copies of the train and test set
train_df = train.copy()
test_df = test.copy()


# In[4]:


train_df.head()


# In[5]:


train_df.info()


# In[6]:


train_df.isnull().sum().sort_values(ascending=False)


# In[7]:


# Safe to say we drop Cabin, Let's fill Age
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())


# In[8]:


train_df["Embarked"].unique()


# In[9]:


train_df["Embarked"].value_counts()


# In[10]:


# Let's fill the nan in Embarked with q
train_df["Embarked"] = train_df[["Embarked"]].apply(lambda x: x.fillna(x.value_counts().index[0]))


# In[11]:


train_df.isnull().sum().sort_values(ascending=False)
# Nice


# In[12]:


# Let's drop Cabin now 
train_df = train_df.drop("Cabin", axis=1)
test_df = test_df.drop("Cabin", axis=1)


# In[13]:


train_df.head()


# ## Feature Selection and EDA
# #### Pclass

# In[14]:


# Let's check each feature and shii, common EDA type shii, Let's start with Pclass
sns.histplot(data=train_df, x = "Pclass")


# In[15]:


sns.barplot(data=train_df, y="Survived", x="Pclass")

# People in class 3 had the lowest chance of survival, so the higher you paid the higher your chance of survival


# In[16]:


# Let's the correlation btw the Survived and Pclass features using spearman's rank correlation, since Pclass is ordianal
from scipy.stats import spearmanr
spearmanr(train_df["Survived"], train_df["Pclass"])


# In[17]:


# They're not strongly correlated but we do see that the different classes have a different effect on the target variable which is good
# To be sure let's use chi-squared test to see the P-value
from scipy.stats import chi2_contingency

# Create a contingency table (cross-tabulation)
contingency_table = pd.crosstab(train_df['Pclass'], train_df['Survived'])

# Perform the chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{expected}")

# yh, let's keep Pclass it's a good feature


# #### Name

# In[18]:


# I think we will drop Name, they are just all unique and shii
train_df = train_df.drop("Name", axis=1)
test_df = test_df.drop("Name", axis=1)


# #### Sex

# In[19]:


# Let's use barplots to check which class, if any, had a higher chance of survival
sns.barplot(data=train_df, x="Survived", y="Sex")


# In[20]:


# We actually see that females survived more, maybe men were self sacrificial and Chilvarous?


# In[21]:


# Let's encode it with OneHotEncoder 
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(train_df[["Sex"]])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(["Sex"]))
train_df = pd.concat([train_df, encoded_df], axis=1)


# In[22]:


train_df.head(3)


# In[23]:


train_df = train_df.drop("Sex", axis=1)


# #### Age

# In[24]:


sns.boxplot(data=train_df, x="Survived", y="Age")


# In[25]:


# Age not looking too crazy rn, let's do point biserial correlation, let's see if it's useful in any way
from scipy.stats import pointbiserialr
pointbiserialr(train_df["Age"], train_df["Survived"])


# In[26]:


# Correlation is crazy bad, the p-value isn't too wild either, let's see ANOVA
from scipy.stats import f_oneway
f_oneway(train_df["Age"], train_df["Survived"])


# In[27]:


# Age not crazy rn, let's view the distro
sns.histplot(train_df["Age"])


# In[28]:


# That's crazily useless, let's try to group the age data 
train_df["AgeGroup"] = pd.cut(
    train_df["Age"],
    bins=[0, 13, 19, 60, np.inf],
    labels=["Child", "Teen", "Adult", "Senior"]
)
# Let's see how each Agegroup relates to the target variable
sns.barplot(data=train_df, x="AgeGroup", y="Survived", errorbar=None)
plt.title("Survival Rate by Age Group")
plt.show()


# In[29]:


# Seems to be some difference, I'll drop Age and use AgeGroup, let's take a chi-squared test for it
contingency_table = pd.crosstab(train_df['AgeGroup'], train_df['Survived'])

# Perform the chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{expected}")

# I guess we'll keep AgeGroup, but let's drop Age


# In[30]:


train_df = train_df.drop("Age", axis=1)


# In[31]:


# Let's encode AgeGroup with OHE 
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(train_df[["AgeGroup"]])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(["AgeGroup"]))
train_df = pd.concat([train_df, encoded_df], axis=1)


# In[32]:


train_df = train_df.drop("AgeGroup", axis=1)


# In[33]:


train_df.head(3)


# #### SibSp

# In[34]:


train_df.SibSp.unique()


# In[35]:


sns.barplot(data=train_df, y="Survived", x="SibSp", errorbar=None)


# In[36]:


train_df["SibSp"].value_counts()


# In[37]:


# let's pop some pointbiserialr on this
pointbiserialr(train_df["Survived"], train_df["SibSp"])


# In[38]:


contingency_table = pd.crosstab(train_df['SibSp'], train_df['Survived'])

# Perform the chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{expected}")


# There's variation in the feature but it's not highly correlated or anything, I might drop it, we'll see


# #### Parch

# In[39]:


train_df["Parch"].value_counts()


# In[40]:


sns.barplot(data=train_df, y="Survived", x="Parch", errorbar=None)


# In[41]:


# let's pop some pointbiserialr on this
pointbiserialr(train_df["Survived"], train_df["SibSp"])


# In[42]:


contingency_table = pd.crosstab(train_df['Parch'], train_df['Survived'])

# Perform the chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{expected}")


# #### Ticket

# In[43]:


train_df["Ticket"].value_counts()
## nahhhhh let's just go ahead and drop this no?


# In[44]:


train_df = train_df.drop("Ticket", axis=1)


# In[45]:


train_df.head(3)


# #### Fare

# In[46]:


# Finally the good stuff
sns.boxplot(data=train_df, x="Survived", y="Fare")


# In[47]:


# Oh it's pretty obvious that people who paid higher have a giher chance of survival, I'm worried, this may be correlated to Pclass, we'll check
# let's check with pearsonsr and spearmanr
from scipy.stats import pearsonr, kruskal
kruskal(train_df["Pclass"], train_df["Fare"]) # Kruskal-wallis test is like ANOVA, but does not assume normality


# In[48]:


spearmanr(train_df["Pclass"], train_df["Fare"])


# In[49]:


sns.boxplot(data=train_df, x="Pclass", y="Fare")

# Seems that yes the Fare prices do affect the class of the ticket


# In[50]:


# Nahh we good, let's check the feature now
sns.histplot(train_df["Fare"])


# In[51]:


# Let's check how Fare correlates to Survived
pointbiserialr(train_df["Survived"], train_df["Fare"])

# There's something there no(loll), gucci tho, I think I'll keep it


# #### Cabin

# In[52]:


# We already dropped it smh


# #### Embarked

# In[53]:


# Just gonna encode that with OHE, let's see a barplot tho
sns.barplot(data=train_df, x="Embarked", y="Survived")

# People who embarked in Cherbourg have a higher chance of survival, chill rq, let's see some corr coeff


# In[54]:


# Let's use cramer's V and then chi-squared test
from scipy.stats.contingency import association
contingency_table2 = pd.crosstab(train_df['Embarked'], train_df['Survived'])
association(contingency_table2)

# Oh no it's kinda low, but I want to make the model and see how each feature helps the model, so we leave it for now


# In[55]:


chi2, p, dof, expected = chi2_contingency(contingency_table2)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{expected}")

# Yup there's variation in the categories


# In[56]:


# Let's encode it with OHE 
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(train_df[["Embarked"]])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(["Embarked"]))
train_df = pd.concat([train_df, encoded_df], axis=1)


# In[57]:


train_df = train_df.drop("Embarked", axis=1)


# In[58]:


train_df.head(3)


# In[59]:


# Let's roll with this for now, we'll see how the features affect the end result and see what we can drop


# ### List of Transformations and the columns they were affected by
# ##### Age - fillna with median
# ##### Cabin, Name, Age, Ticket - Dropped 
# ##### Sex, Embarked - OneHotEncoded
# ##### AgeGroup -  created from Age

# # Modeling

# In[60]:


# ok Let's start making our logistic regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# In[61]:


# Getting the X and y dataframes
X = train_df.drop(["Survived", "PassengerId"], axis=1)
y = train_df["Survived"]


# In[62]:


# Just getting the train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)


# In[63]:


log = LogisticRegression(n_jobs=-1, max_iter=1000)


# In[64]:


log.fit(X_train, y_train)


# In[65]:


y_pred = log.predict(X_test)


# In[66]:


# Let's see the accuracy of the model
accuracy_score(y_test, y_pred)


# ### Model Tuning

# In[98]:


# Let's do feature Importances to see how important each feature is to out model's final outcome
coefficients = log.coef_[0]
odds_ratios = np.exp(coefficients)


# Display feature importance using coefficients and odds ratios
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'Odds Ratio': odds_ratios
})
print("\nFeature Importance (Coefficient and Odds Ratio):")
print(feature_importance.sort_values(by='Coefficient', ascending=False))


# In[68]:


# Recursive Feature Elimination(RFE), we make a new model which recursively checks for the best features to be used for modeling
from sklearn.feature_selection import RFE
rfe_model = LogisticRegression(max_iter=10000, solver='liblinear')
rfe = RFE(rfe_model, n_features_to_select=8)
rfe.fit(X_train, y_train)


rfe_features = X.columns[rfe.support_]
print("\nSelected Features by RFE:")
print(rfe_features)


# In[74]:


# Ok I wanna make a model with L1 regularization
l1_model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, C=1.0)


# In[75]:


l1_model.fit(X_train, y_train)


# In[76]:


new_preds = l1_model.predict(X_test)


# In[79]:


# Let's do cross_val on the model
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(l1_model, X, y, cv=5, scoring='accuracy')


# In[80]:


# Now let's calculate the mean CV score
mean_cv_score = cv_scores.mean()

print("Average 5-fold cross-validation score:", mean_cv_score)


# In[87]:


# I'm dead, how is it worse, well for our final magic trick, let's do GridSearchCV
# Firstly let's make a param grid ourself, I'm learning so much fr fr
from sklearn.model_selection import GridSearchCV
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [.25, .50, .75]
}

grid_search = GridSearchCV(log, param_grid, scoring='accuracy', n_jobs=-1)


# In[88]:


grid_search.fit(X_train, y_train)


# In[92]:


# Let's getthe best regularization strength and penalty
print("Best regularization strength:", grid_search.best_params_['C'])
print("Best penalty:", grid_search.best_params_['penalty'])

if grid_search.best_params_['penalty'] == 'elasticnet':
    print("Best alpha:", grid_search.best_params_['l1_ratio'])


# In[93]:


best_model = LogisticRegression(C=100.0, max_iter=10000, n_jobs=-1)


# In[94]:


best_model.fit(X_train, y_train)


# In[95]:


y_best = best_model.predict(X_test)


# In[96]:


accuracy_score(y_test, y_best)
# This is basically the best accuracy we can get


# In[ ]:


# Ok so now I know my best features & best model params, let's just pipeline everything for the data set and make the best model


# ### List of Transformations and the columns they were affected by
# ##### Age - fillna with median
# ##### Cabin, Name, Age, Ticket - Dropped 
# ##### Sex, Embarked - OneHotEncoded
# ##### AgeGroup -  created from Age

# In[97]:


# Let's stop here for today and make notes of what to do when we pick up next time

Sex_female
AgeGroup_Child
Pclass
Fare.
Embarked_C and Embarked_Q: 
SibSp
AgeGroup_Senior

I might exclude:
Sex_male
Embarked_S
# In[101]:


# Let's make a new model with these features above and see if it's accuracy gets higher
X_main = X.drop(["Parch", "AgeGroup_Adult"], axis=1)
X_main_train, X_main_test, y_main_train, y_main_test = train_test_split(X_main, y, random_state=42, test_size=0.3)


# In[103]:


new_model = LogisticRegression(C=100.0, max_iter=10000, n_jobs=-1)
new_model.fit(X_main_train, y_main_train)


# In[105]:


y_main_pred = new_model.predict(X_main_test)


# In[106]:


accuracy_score(y_main_test, y_main_pred)


# In[109]:


# It's still worse, this is crazy, chill rq let's go again
X_m = X.drop(["Parch", "AgeGroup_Adult", "Sex_male", "Embarked_S"], axis=1)
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_m, y, random_state=42, test_size=0.3)


# In[110]:


new_model = LogisticRegression(C=100.0, max_iter=10000, n_jobs=-1)
new_model.fit(X_m_train, y_m_train)


# In[111]:


y_m_pred = new_model.predict(X_m_test)


# In[112]:


accuracy_score(y_m_test, y_m_pred)


# In[ ]:


# We actually can't get anything better than the base log.reg model crazyyyy, ok let's keep going


# # Pipelining and final model

# ## What to do
# #
# #### 1. Pipeline everything I did
# #### 2. Do not drop anything from the final X features, it doesn't improve model performance