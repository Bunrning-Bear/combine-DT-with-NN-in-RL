#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.21
# Modified    :   2017.6.21
# Version     :   1.0
import numpy as np

def onehot_encoder(train_path,test_path,onehot_location,class_amount):
    train_data = open("../../data/data/train.csv","r")  
    test_data = open("../../data/data/test.csv","r") 
    ##train data  
    train_feature=[]  
    train_target=[]  
    for line in train_data:  
        temp = line.strip().split(',')  
        train_feature.append(map(int,temp[0:-1]))  
        train_target.extend(map(int,temp[-1]))  
    train_data.close()  
    test_feature=[]  
    test_target=[]      
    for line in test_data:  
        temp = line.strip().split(',')  
        test_feature.append(map(int,temp[0:-1]))  
        test_target.extend(map(int,temp[-1]))  
    test_data.close()  
    train_feature = np.array(train_feature)  
    test_feature = np.array(test_feature)  
    ##OneHotEncoder used  
    enc = OneHotEncoder(
        categorical_features=np.array(onehot_location),
        n_values=class_amount)  
    enc.fit(train_feature)  
    train_feature = enc.transform(train_feature).toarray()  
    test_feature = enc.transform(test_feature).toarray()  
    clf = RandomForestClassifier(n_estimators=10)  
    clf = clf.fit(train_feature,train_target)  