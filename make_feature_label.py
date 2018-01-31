import numpy as np
import pandas as pd
import math

def make_feature_label(f_data,l_data,train_user,train=True):
    if train==False:
        f_data_merge=f_data[['user_id','item_id']].drop_duplicates()
        test_xx=f_data_merge
        
        
    else:

        f_data_drop=f_data[['user_id','item_id']].drop_duplicates()

        f_data_merge=pd.merge(f_data_drop, l_data[['user_id','item_id','behavior_type']],\
                           on=['user_id', 'item_id'],how='left').fillna(0)
        print('aaaaaaaaaaaaaa',f_data_merge)

        f_data_merge.behavior_type= f_data_merge.behavior_type.map(lambda x:1 if x==4 else 0)

        f_data_merge=f_data_merge.groupby(['user_id','item_id']).sum()   

        print('f_data_merge_sum',f_data_merge.head(5))


        f_data_merge=f_data_merge.reset_index()


        f_data_merge.behavior_type=f_data_merge.behavior_type.apply(lambda x:1 if x>0 else 0)

        f_data_merge.rename(columns={'behavior_type':'label'}, inplace = True)


    for b_type in [1,2,3,4]:
        b_havior=f_data[f_data.behavior_type==b_type][['user_id','item_id','behavior_type']]\
                            .groupby(['user_id','item_id']).count().reset_index()
        b_havior.rename(columns={'behavior_type':'behavior_type'+str(b_type)+'_count',}, inplace = True)
        f_data_merge=pd.merge(f_data_merge,b_havior,on=['user_id', 'item_id'],how='left')
        
        
    all_type4=train_user[train_user.behavior_type==4][['user_id','behavior_type']]
    all_type4=all_type4.groupby(['user_id']).count().reset_index()
    all_type4.rename(columns={'behavior_type':'usr_shop_count'}, inplace = True)
    f_data_merge=pd.merge(f_data_merge,all_type4,on=['user_id'],how='left')
    
    #添加畅销品系数
    all_type4=train_user[train_user.behavior_type==4][['item_id','behavior_type']]
    all_type4=all_type4.groupby(['item_id']).count().reset_index()
    all_type4.rename(columns={'behavior_type':'item_id_count'}, inplace = True)
    f_data_merge=pd.merge(f_data_merge,all_type4,on=['item_id'],how='left')
    
    f_data_merge=f_data_merge.fillna(0)
    feature=f_data_merge[[x for x in f_data_merge.columns if x not in ['user_id', 'item_id', 'label']]]
    
    #做一下平滑，将数据范围缩小
    for col in feature.columns:
        feature[col]=feature[col].apply(lambda x:math.log1p(x))

    if train==True:
        label=f_data_merge['label']
        return feature,label
    else:
        return feature,test_xx



