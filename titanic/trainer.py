import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ctx = 'C:/Users/ezen/PycharmProjects/day1/titanic/data/'
train = pd.read_csv(ctx + 'train.csv')
test = pd.read_csv(ctx + 'test.csv')
df = pd.DataFrame(train)
# print(train.head())
# print(df.columns)
"""
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
PassengerId 고객 ID
Survived 생존여부   Survival    0 = No, 1 = Yes
Pclass 승선권 클래스   Ticket class    1 = 1st, 2 = 2nd, 3 = 3rd
Name 이름Sex 성별   Sex    
Age 나이   Age in years    
SibSp 동반한 형제자매, 배우자 수   # of siblings / spouses aboard the Titanic    
Parch 동반한 부모, 자식 수   # of parents / children aboard the Titanic    
Ticket 티켓 번호   Ticket number    
Fare 티켓의 요금   Passenger fare    
Cabin 객실번호   Cabin number    
Embarked 승선한 항구명   Port of Embarkation   
    C = Cherbourg, Q = Queenstown, S = Southampton
"""

f, ax = plt.subplots(1, 2, figsize=(18, 8))
train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct="%1.1f%%", ax=ax[0], shadow=True)

ax[0].set_title('Survived')
ax[0].set_ylabel('')

sns.countplot('Survived', data=train, ax=ax[1])
ax[1].set_title('Survived')

# plt.show() 생존률 38.4% 사망률 61.6%

"""
데이터는 훈련데이터(train.csv), 목적데이터(test.csv) 두가지로 제공됩니다.
목적데이터에는 위 항목에서는 Survived 정보가 바져있습니다.
그것은 답이기 때문입니다.
"""

# **********
# 성별
# **********

f, ax = plt.subplots(1, 2, figsize=(18, 8))
train['Survived'][train["Sex"] == 'male'].value_counts().plot.pie(explode=[0, 0.1], autopct="%1.1f%%", ax=ax[0], shadow=True)
train['Survived'][train["Sex"] == 'female'].value_counts().plot.pie(explode=[0, 0.1], autopct="%1.1f%%", ax=ax[1], shadow=True)

ax[0].set_title('Survived(Male)')
ax[1].set_title('Survived(Female)')

# plt.show()
# 남성 생존율 18.9% 사망률 81.1%
# 여성 생존율 74.2% 사망률 25.8%


# **********
# 승선권
# **********

df_1 = [train['Sex'], train['Survived']]
df_2 = train['Pclass']
df = pd.crosstab(df_1, df_2, margins=True)
# print(df)

"""
Pclass             1    2    3  All
Sex    Survived                    
female 0           3    6   72   81
       1          91   70   72  233
male   0          77   91  300  468
       1          45   17   47  109
All              216  184  491  891
"""

f, ax = plt.subplots(2, 2, figsize=(20, 15))
sns.countplot('Embarked', data=train, ax=ax[0, 0])
ax[0, 0].set_title('No. Of Passengers Pclass')
sns.countplot('Embarked', hue='Sex', data=train, ax=ax[0, 1])
ax[0, 1].set_title('Male - Female Pcalss')
sns.countplot('Embarked', hue='Survived', data=train, ax=ax[1, 0])
ax[1, 0].set_title('Pclass vs Survived')
sns.countplot('Pclass', data=train, ax=ax[1, 1])
ax[1, 1].set_title('No. Of Passengers Pclass')

# plt.show()

"""
위 데이터를 보면 절반 이상의 승객이 ‘Southampton’에서 배를 탔으며, 여기에서 탑승한 승객의 70% 가량이 남성이었습니다. 현재까지 검토한 내용으로는 남성의 사망률이 여성보다 훨씬 높았기에 자연스럽게 ‘Southampton’에서 탑승한 승객의 사망률이 높게 나왔습니다.
또한 ‘Cherbourg’에서 탑승한 승객들은 1등 객실 승객의 비중 및 생존률이 높은 것으로 보아서 이 동네는 부자동네라는 것을 예상할 수 있습니다.
"""

#결측치 제거
# print(train.info())
"""
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
"""

# print(train.isnull().sum())
"""
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
"""

def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['survived', 'dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
    plt.show()


# bar_chart('Sex')
# bar_chart('Pclass')
# bar_chart('SibSp')
# bar_chart('Parch')
# bar_chart('Embarked')

train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)

s_city = train[train['Embarked']=='S'].shape[0]
c_city = train[train['Embarked']=='C'].shape[0]
q_city = train[train['Embarked']=='Q'].shape[0]

# print("S = ", s_city) #644
# print("C = ", c_city) #168
# print("Q = ", q_city) #77

train = train.fillna({"Embarked":"S"})
s_city = train[train['Embarked']=='S'].shape[0]
# print("S = ", s_city) #644

city_mapping = {"S":1, "C":2, "Q":3}
train['Embarked'] = train['Embarked'].map(city_mapping)
test['Embarked'] = test['Embarked'].map(city_mapping)

# print(train.head())
# print(test.head())

combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
# print(pd.crosstab(train['Title'], train['Sex']))

for dataset in combine:
    dataset['Title'] \
        = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] \
        = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] \
        = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] \
        = dataset['Title'].replace('Mme', 'Mrs')
# print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
'''
    Title  Survived
0  Master  0.575000
1    Miss  0.701087
2      Mr  0.156673
3     Mrs  0.793651
4      Ms  1.000000
5    Rare  0.250000
6   Royal  1.000000
'''

train = train.drop(['Name', 'PassengerId'], axis=1)
test= test.drop(['Name', 'PassengerId'], axis=1)
combine = [train, test]
# print(train.head())

sex_mapping = {"male":0, "female":1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

# Age 가공하기
train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels=labels)

# print(train.head())

# print (train['Title'])

# print(train['Title'])

title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5,'Rare':6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0) # fillna

age_mapping = {'Baby' : 1, 'Child':2, 'Teenager':3, 'Student': 4, 'Young Adult':5,'Adult':6, 'Senior':7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = train['AgeGroup'].map(age_mapping)
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)
# print(train['Title'])

age_title_mapping = {1: "Young Adult", 2:"Student", 3:"Adult", 4:"Baby", 5:"Adult", 6:"Adult"}
for x in range(len(train['AgeGroup'])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train['Title'][x]]
for x in range(len(test['AgeGroup'])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test['Title'][x]]
# train.head()


train['FareBand'] = pd.qcut(train['Fare'], 4, labels = {1,2,3,4})
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = {1,2,3,4})


train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)
train.head()

train_data = train.drop('Survived', axis = 1)
target = train['Survived']
print(train_data.shape, target.shape)