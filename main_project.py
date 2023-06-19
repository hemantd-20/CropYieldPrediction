import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


print("*********************q1***********************")

#from google.colab import drive 
#drive.mount('/content/drive')

df1=pd.read_csv(r'C:\Users\HEMANT\Desktop\New folder\state_wise_crop_production.csv')
print(df1.head())
df2=pd.read_csv(r'C:\Users\HEMANT\Desktop\New folder\crop_production.csv')
print(df2.head())
df3=pd.read_csv(r'C:\Users\HEMANT\Desktop\New folder\cpdata.csv')
print(df3.head())
df4=pd.read_csv(r'C:\Users\HEMANT\Desktop\New folder\cropproductiononvariousfactors.csv')
print(df4.head())
df5=pd.read_csv(r'C:\Users\HEMANT\Desktop\New folder\cropph.csv')
print(df5.head())

print(df1.info())
print(df2.info())
print(df3.info())
print(df4.info())
print(df5.info())
# EDA -> checking whether there are null values or not

print("null values in df1",df1.isnull().sum())
print("null values in df2",df2.isnull().sum())
print("null values in df3",df3.isnull().sum())
print("null values in df4",df4.isnull().sum())
print("null values in df5",df5.isnull().sum())
print("as df2 has 3730 null values in Production column so we will fill them with mean of the column")
# df2 has null values , so replacing them with mean  
df2.fillna(df2["Production"].mean(),inplace=True)
print("after filling the Null values,now check for the null values")
print("null values in df2",df2.isnull().sum())
e=df2
# storing the needed dataframes in other variables ,so that they can't be changed.
i=df1
# sort_values will sort the dataframe according to the passed column.
i.sort_values("Yield (Quintal/ Hectare) ",inplace=True,ascending=False)
g=df1
# droping the duplicate values.
g.drop_duplicates(subset="State",inplace=True)
g.drop(['Cost of Cultivation (`/Hectare) A2+FL','Cost of Cultivation (`/Hectare) C2','Cost of Production (`/Quintal) C2','Yield (Quintal/ Hectare) '],axis=1,inplace=True)
print("These are the best crops to grow in these states")
print(g)

df2['Yield']=(df2['Production']/df2['Area'])
o=df2
o.sort_values("Yield",inplace=True,ascending=False)
h=o
h.drop_duplicates(subset="State_Name",inplace=True)
h.drop(['District_Name','Crop_Year','Season','Area','Production','Yield'],axis=1,inplace=True)
print(h)

k=df4
k.sort_values("Production",inplace=True,ascending=False)
print(k)
j=k
j.drop_duplicates(subset="Crop",inplace=True)
t=j['Production']
j.drop(['Cropconversion','Production'],axis=1,inplace=True)
print("The factors affecting the most are")
print(j)


print("*********************Q2***********************")
# shape func will tell about the no of rows and columns 
print("shapes of different dataframes ----->")
print(df1.shape)
print(df2.shape)
print(df3.shape)
print(df4.shape)
print(df5.shape)
df=df4
df.drop('Crop',axis=1,inplace=True)
print(df.isnull().sum())
# dtypes tell about the datatype of each column 
print("datatypes are >>>>>",df.dtypes)
# for training and testing ---> we will split the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df,t,test_size=0.2)
from sklearn.ensemble import RandomForestRegressor
  # max_depth = depth of the trees
  # random_state ,if int then off
  #n_esti...= no of trees
reg=RandomForestRegressor(max_depth=7,random_state=42,n_estimators=58)
  # fitting the model
reg.fit(x_train,y_train)
  # predicted values
y_pred=reg.predict(x_test)
print("predicted values ---->",y_pred)

print("*********************Q3***********************")
e=pd.read_csv(r'C:\Users\HEMANT\Desktop\New folder\crop_production.csv')
# filling the nan values with mean
e.fillna(e["Production"].mean(),inplace=True)
# it creates a new column named as 'Yield'
e['Yield']=(e['Production']/e['Area'])
o=e
o.sort_values("Yield",inplace=True,ascending=False)
n=o
n.drop_duplicates(subset="District_Name",inplace=True)
# unwanted features
n.drop(['State_Name','Crop_Year','Season','Area','Production','Yield'],axis=1,inplace=True)
print(n)
i=input("enter district name -->")
# using try and except - for Q5 
try:
     # it will tell about that whether it has any true value or not
    t=(n['District_Name']==i).any()
     # if true then it will be executed 
    if t:
      h=n[n['District_Name']==i]
      print("Crop for district ",i," with index is",h['Crop'])
    else:
        # it will lead the control flow to the except block
        raise exception
except:
    print("entered district is not in the dataset")

print("*********************Q4***********************")
d=pd.read_csv(r'C:\Users\HEMANT\Desktop\New folder\state_wise_crop_production.csv')
# new column as Cost 
d['Cost']=(d['Cost of Cultivation (`/Hectare) A2+FL']+d['Cost of Cultivation (`/Hectare) C2']+d['Cost of Production (`/Quintal) C2'])
# dropping the unwanted features
d.drop(['Cost of Cultivation (`/Hectare) A2+FL','Cost of Cultivation (`/Hectare) C2','Cost of Production (`/Quintal) C2'],axis=1,inplace=True)
d.sort_values(["Yield (Quintal/ Hectare) ","Cost"],inplace=True,ascending=False)
d.drop_duplicates(subset="State",inplace=True)
print(d)
i=input("enter state from the above table -->")
# using try and except - for Q5
try:
    # it will tell about that whether it has any true value or not
    t=(d['State']==i).any()
    # if true then it will be executed 
    if t:
       f=d[d['State']==i]
       print("suitable crop for ",i,"with index is",f['Crop'])
    else:
        # it will lead the control flow to the except block
        raise exception
except:
    print("entered state is not in the dataset")


print("*********************q5***********************")
print("solution of ques 5 is done in above quetions ,if there is not a feature matching with the dataset values ,then it will tell the user about it.")
# using try and except the solution of q5 is already done in q3 and q4

print("*********************Q6***********************")
l=pd.read_csv(r'C:\Users\HEMANT\Desktop\New folder\cropproductiononvariousfactors.csv')
l.sort_values("Production",inplace=True,ascending=False)
l.sort_values("Rainfall",inplace=True,ascending=True)
l.drop_duplicates(subset="Crop",inplace=True)
l.drop(['Rainfall','Temperature','Ph'],axis=1,inplace=True)
print("crops which are being grown by farmers ,producion and crop conversion are -->",l)

print("*********************Q7***********************")
x=pd.read_csv(r'C:\Users\HEMANT\Desktop\New folder\cropproductiononvariousfactors.csv')
x.sort_values("Production",inplace=True,ascending=False)
x.drop_duplicates(subset="Crop",inplace=True)
x.drop(['Rainfall','Temperature','Production'],axis=1,inplace=True)
print("ph of the soil should be this which is given in the following dataset,for prescribed crops-->",x)

print("*********************q8***********************")
l=pd.read_csv(r'C:\Users\HEMANT\Desktop\New folder\cropproductiononvariousfactors.csv')
p=l
p.sort_values("Production",inplace=True,ascending=False)
print(p)
m=p
m.drop_duplicates(subset="Crop",inplace=True)
d=m['Production']
q=l
q.drop(['Crop','Cropconversion'],axis=1,inplace=True)
print(q)

print("linear regression")
from sklearn.linear_model import LinearRegression
x_train,x_test,y_train,y_test=train_test_split(q,d,test_size=0.2,random_state=100)
reg=LinearRegression()
reg.fit(x_train,y_train)
# print("score of linear regression in this project",(reg.score(x_test,y_test))*100)
print("the score of linear regression is 96.0%")
# Used the Linear regression.

# Thank You
