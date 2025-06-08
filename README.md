# 🚢 Titanic Survival EDA & Prediction Model

**Author**: Mythrre Thota  
**Date**: 8 June 2025  
**Dataset**: [Titanic – Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic/data)

---

**Objective**

The Titanic dataset is a classic classification problem that predicts survival of passengers based on features like age, gender, passenger class, fare, and family size.

This notebook performs:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Survival rate visualizations
- Analysis of gender, class, and age impact on survival
- Machine learning model comparisons for prediction

---

**Dataset Loading**

```python
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df.head()
```


**Data Cleaning**
```python
# Fill missing Age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column due to many missing values
df.drop(['Cabin'], axis=1, inplace=True)
```
**Survival by Gender**
Females had significantly higher survival rates than males.
This supports the notion that “women and children first” was practiced during evacuation.

**Survival by Age Groups**
```python
Copy
Edit
df['Agebin'] = pd.cut(df['Age'], bins=[0,10,20,30,40,50,60,80], labels=['0-10','10-20','20-30','30-40','40-50','50-60','60+'])
```

**Age Bin	Survival Rate (%)**
0-10	59.38
10-20	38.26
20-30	33.42
30-40	44.52
40-50	38.37
50-60	40.48
60+	22.73

Children under 10 had the highest survival rate, likely prioritized for rescue.
Survival rates drop significantly for older passengers.
Survival by Passenger Class (Pclass)

**Pclass	Survival Rate (%)**
1	62.96
2	47.28
3	24.24

Passengers in 1st class had the highest survival rates, possibly due to better cabin locations and access to lifeboats.


**Survival by Gender & Class**

Class	Gender	Survival Rate (%)
1	Female	96.81
1	Male	36.89
2	Female	92.11
2	Male	15.74
3	Female	50.00
3	Male	13.54

Gender strongly influenced survival within every class.
Females had a much higher chance of survival than males across all classes.

**Impact of Family Size on Survival**
```python
Copy
Edit
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Alone'] = (df['FamilySize'] == 0)
```
**Condition	Survival Rate (%)**
Alone	30.35
With Family	50.56

Passengers traveling with family had a significantly better chance of survival.

Survival Rate by Number of Siblings/Spouses (SibSp)
SibSp	Survival Rate (%)
0	34.54
1	53.59
2	46.43
3	25.00
4	16.67
5	0.00
8	0.00

Survival rates peaked when passengers had 1 or 2 siblings/spouses aboard.

Larger family groups had lower survival rates.

Combined Analysis: Class, Gender & Age Group
Class	Gender	Age Bin	Survival Rate (%)
1	Female	20-30	96.67
1	Male	20-30	35.00
3	Female	20-30	55.41
3	Male	20-30	12.04

Highest survival rates were seen in young and middle-aged females in 1st and 2nd classes.
Older males in 3rd class had some of the lowest survival rates.

**Next Steps**
Train machine learning models to predict passenger survival
Compare model performance and accuracy

**References**
Kaggle Titanic Dataset


Author: Mythrre Thota
Date: 8 June 2025
---










