import pandas as pd
from make_feature_label import make_feature_label
import numpy as np
import math
import datetime 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
train_item= pd.read_csv("data/tianchi_fresh_comp_train_item.csv")
train_user = pd.read_csv("data/tianchi_fresh_comp_train_user.csv")
train_item.head()
train_user.info()
train_user.head()

train_user.time= train_user.time.map(lambda time:time.split()[0]) #？？？？？？？？？
print(train_user.time.head(4))
train_user.head()


date_map={}  #定义一个空字典
begin = datetime.date(2014,11,18)  #datetime.date：表示日期的类。常用的属性有year, month, day；
end = datetime.date(2014,12,18)  
d = begin  
delta = datetime.timedelta(days=1)  #每天增加一天
while d <= end:  
    date_map[d.strftime("%Y-%m-%d")]=len(date_map)   #字典有长度，将长度赋给字典
    #time.strftime()可以用来获得当前时间，可以将时间格式化为字符串，用
    d += delta  
print(date_map)

train_user.time=train_user.time.map(date_map)
train_user.head()

## test :30取特征--31预测提交
data_28=train_user[train_user.time==28]
data_29=train_user[train_user.time==29]
data_30=train_user[train_user.time==30]

print(data_28.head(),data_28.shape)


    
    
train_x,train_y=make_feature_label(data_28,data_29,train_user,True)
print('aaaaaaa')
dev_x,dev_y=make_feature_label(data_29,data_30,train_user,True)
print('bbbbbbbb')
test_x,test_xx=make_feature_label(data_30,'',train_user,False)




model= LogisticRegression()
model.fit(train_x,train_y)
preds=model.predict(dev_x)
print(model.score(dev_x,dev_y))
print(accuracy_score(dev_y,preds))
print(f1_score(dev_y,preds))
n_s=test_x.shape[0]
log_prob=model.predict_log_proba(test_x)[:,1]
#print(sorted((,reverse=True)[:100])

idx_s=(np.argsort(-log_prob)[:2000])
def filter_(x):
    if x in idx_s:
        return True
    else :
        return False
idx=[filter_(x) for x in range(n_s)]
test_xx=test_xx.reset_index()[['user_id','item_id']]
print(test_xx.head())

#rest=[test_xx[x] for x in idx]

test_xx[idx].to_csv("tianchi_mobile_recommendation_predict.csv",header=False,index=False)



