import sys
import numpy as np
import pandas as pd
import csv
print(sys.argv[1],sys.argv[2],sys.argv[3])
raw_data = np.genfromtxt(sys.argv[1], delimiter=',', encoding = 'big5')
data = raw_data[1:,3:]   # 4320 * 24   , 24hr , 4320= 18-dim * 12months * 20days
where_are_NaNs = np.isnan(data)
data[where_are_NaNs] = 0 
month_to_data = {}  ## Dictionary (key:month , value:data)                                  
for month in range(12):
    sample = np.empty(shape = (18 , 480)) # sample : every month data :18-dim * 480hrs (20days * 24hr)
    for day in range(20):
        for hour in range(24): 
            sample[:,day * 24 + hour] = data[18 * (month * 20 + day): 18 * (month * 20 + day + 1),hour]
    month_to_data[month] = sample    # month_to_data : 12 months * 18-dim * 480 
    
x = np.empty(shape = (12 * 471 , 18 * 9),dtype = float) # the prvious 9 hour as model input ,ans concate,so 18*9
y = np.empty(shape = (12 * 471 , 1),dtype = float)  # the 10th hour as model target

for month in range(12): 
    for day in range(20): 
        for hour in range(24):   
            if day == 19 and hour > 14:
                continue  
            x[month * 471 + day * 24 + hour,:] = month_to_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1,-1) 
            #  * 18-dim * 9 
            y[month * 471 + day * 24 + hour,0] = month_to_data[month][9 ,day * 24 + hour + 9]
train_x = (x-np.mean(x,axis=0))/np.std(x,axis=0)
train_xx = x/np.max(x,axis =0)

#mean = np.mean(x, axis = 0) 
#std = np.std(x, axis = 0)
#for i in range(x.shape[0]):
 #   for j in range(x.shape[1]):
  #      if not std[j] == 0 :
   #         train_x[i][j] = (train_x[i][j]- mean[j]) / std[j]
    
    
dim = x.shape[1] + 1 
#w = np.zeros(shape = (dim, 1 ))
w = np.random.normal(loc =0 ,scale = np.sqrt(1/train_x.shape[1]),size = (dim,1))
X = np.concatenate((np.ones((train_x.shape[0], 1 )), train_x) , axis = 1).astype(float)
#learning_rate = np.array([[200]] * dim)
adagrad_sum = np.zeros(shape = (dim, 1 ))

# shuffle 
shuffle = np.concatenate( (X,y) ,axis=1) 
np.random.shuffle( shuffle)
X = shuffle[:,:-1]
y = shuffle[:,-1].reshape(-1,1)


for T in range(10000): # no batch , taining all data 
    learning_rate = np.array([[1* (0.5**int(T/1000))]] * dim )
    #learning_rate = np.array([[200]] * dim)
    if(T % 500 == 0 ):
        print("T=",T)
        print("Loss:",np.sum(np.power(X.dot(w) - y, 2 ))/ X.shape[0] )
        shuffle = np.concatenate( (X,y) ,axis=1) 
        np.random.shuffle( shuffle)
        X = shuffle[:,:-1]
        y = shuffle[:,-1].reshape(-1,1)

        
    ''' 
    #without batch    
    gradient = (-2) * np.transpose(X).dot(y-X.dot(w))
    adagrad_sum += gradient ** 2
    w = w - learning_rate * gradient / (np.sqrt(adagrad_sum) + 0.0005)
    '''
    
    
    # batch
    
    for j in range(12):
        gradient = (-2) * np.transpose(X[471*j:471*(j+1)]).dot(y[471*j:471*(j+1)]-X[471*j:471*(j+1)].dot(w))
        adagrad_sum += gradient ** 2
        w = w - learning_rate * gradient / (np.sqrt(adagrad_sum) + 0.0005)
    
    
    
    

np.save('weight.npy',w)     ## save weight
w = np.load('weight.npy')                                   ## load weight
test_raw_data = np.genfromtxt(sys.argv[2], delimiter=',' ,encoding = 'big5')   ## test.csv
test_data = test_raw_data[:, 2: ]
where_are_NaNs = np.isnan(test_data)
test_data[where_are_NaNs] = 0 
test_x = np.empty(shape = (240, 18 * 9),dtype = float)
for i in range(240):
    test_x[i,:] = test_data[18 * i : 18 * (i+1),:].reshape(1,-1) 
    
test_x = (test_x-np.mean(x,axis=0))/np.std(x,axis=0)
test_x = np.concatenate((np.ones(shape = (test_x.shape[0],1)),test_x),axis = 1).astype(float)
answer = test_x.dot(w)
f = open(sys.argv[3],"w")
w = csv.writer(f)
title = ['id','value']
w.writerow(title) 
for i in range(240):
    content = ['id_'+str(i),answer[i][0]]
    w.writerow(content) 