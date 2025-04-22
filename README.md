## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd

df=pd.read_csv("/content/Encoding Data (2).csv")

df
```
![image](https://github.com/user-attachments/assets/66b2270c-6072-4bbf-bc9c-f296712cf190)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/b96dc0a3-ab2a-40ca-9f77-ec1fac4a0869)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])

df
```
![image](https://github.com/user-attachments/assets/34ef4127-7d45-43bf-b45c-626be855682e)
```
le=LabelEncoder()

dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2'])

dfc
```
![image](https://github.com/user-attachments/assets/b9c73cf6-602b-4e66-88e4-04b152c71085)
```
from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1)

df2
```
![image](https://github.com/user-attachments/assets/7ad1756d-01f1-48e6-b045-fc2ce6411d3a)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/11633f8a-67e3-4588-a90a-0a21d801b146)
```
pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

df=pd.read_csv("/content/data (2).csv")

df
```
![image](https://github.com/user-attachments/assets/aa6906c1-d6c7-4dbe-9f4c-5b91acbc071e)
```
be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])

dfb=pd.concat([df,nd],axis=1)

dfb1=df.copy()

dfb
```
![image](https://github.com/user-attachments/assets/b2d33752-9963-461a-8ad5-38e1fb7c1fb4)
```
from category_encoders import TargetEncoder

te=TargetEncoder()

cc=df.copy()

new=te.fit_transform(X=cc["City"],y=cc["Target"])

cc=pd.concat([cc,new],axis=1)

cc
```
![image](https://github.com/user-attachments/assets/ff3a101b-c307-46e0-95a7-060f2c45940b)
```
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("/content/Data_to_Transform (1).csv")

df
```
![image](https://github.com/user-attachments/assets/27930535-b647-419d-9b94-3534936a78cc)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/c35dd008-0d19-4e2c-b592-03dc877e8aaf)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/70896f0b-9c29-4c9b-9c8e-24f832b707f6)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/dab86fc4-1b53-47c2-9a58-e3bdc01474b3)
```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/d05c5f2b-fe3f-447f-92dd-56576ba08d0e)
```
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/430ec66b-80a8-4179-abef-ab32ba3d0c7f)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])

df
```

![image](https://github.com/user-attachments/assets/9e327c91-b03c-47a3-8788-4c6dd061a6a5)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])

df.skew()
```

![image](https://github.com/user-attachments/assets/1cb1f257-0702-4bd3-be0b-eba7265e331d)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])

df.skew()
```

![image](https://github.com/user-attachments/assets/ffe8f80b-2629-4b25-bb7c-783b35815394)
```
from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])

df
```

![image](https://github.com/user-attachments/assets/17288f3b-777b-4f40-934c-00ca195c3074)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/5e8f87ea-1232-4374-a222-97263a227e50)
```
sm.qqplot(np.reciprocal(df["Highly Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/9e550fac-5285-4593-949b-ebbf2a7f2c49)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/b0a68180-125f-4b6a-befb-d1c45ae98b53)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/17d9c2e4-e08f-4465-9ec8-b982f1a4b576)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/27cea452-472c-4fb7-bd9e-c5b5e3c48eab)
```
dt=pd.read_csv("/content/titanic_dataset (2).csv")

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[['Age']])

sm.qqplot(dt['Age'],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/9860557e-da49-4f34-bc92-0a4f92c187be)
```
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/7e3f9c7a-764f-46ff-83b6-765c36ea9543)

# RESULT:
     Successfully read the given data and perform Feature Encoding and Transformation process and save the data to a file.  

       
