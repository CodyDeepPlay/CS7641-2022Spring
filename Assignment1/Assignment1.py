#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 21:12:55 2022

@author: mingmingzhang
"""



#%% load some data

import os
import matplotlib.pyplot as plt

import gzip
import numpy as np
import pandas as pd

# classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler

#% Below are two separate datasets

###############################################################################
###        MINIST datasets, hand writing digits images               ##########
###############################################################################
### downloaded from  http://yann.lecun.com/exdb/mnist/
### the training dataset contain 60,000 examples


image_size = 28
num_images = 60000


# load each training examples, an individual images
f = gzip.open('data/train-images-idx3-ubyte.gz','r')
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
digit_img = data.reshape(num_images, image_size, image_size, 1)

# load the related training labels
f = gzip.open('data/train-labels-idx1-ubyte.gz','r')
labels = []
f.read(8)
for i in range(0,num_images):   
    buf = f.read(1)
    one_label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)[0]
    labels.append(one_label)
    
    
# # plot one examples to see
# plt.figure()
# n = 6001
# plt.imshow(digit_img[n,:,:,:], cmap=plt.cm.gray_r,)
# plt.title('This is ' + str(labels[n]))

# select 16000 examples to conduct our experiment in this assignment
selected_length = 10000
img_data   = digit_img[0:selected_length].reshape(selected_length, image_size*image_size)
img_labels = labels[0:selected_length]

print('Total number of images selected:')
print(len(img_data))

X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(img_data, img_labels, test_size=0.2, random_state=42)

X = StandardScaler().fit_transform(img_data)
X_train_img_standard, X_test_img_standard, y_train_img_standard, y_test_img_standard = train_test_split(X, img_labels, test_size=0.2, random_state=42)



# investigate the number of classes in each case.
num_instances = []  # store the number of instances for each class
total_class = list(set(y_train_img)) # figure out how many classes are there in the training data
y_train_img_array = np.asarray(y_train_img)
for a_class in total_class:
    num_instances.append( len(np.where(y_train_img_array == a_class)[0]) )
    

# plot a figure to show the number of instances in each class in the training data
#plt.figure()
fig, ax = plt.subplots()
p1 = ax.bar(total_class, num_instances)
plt.ylabel('Number of instances')
plt.title('Data distribution across different classes in MINIST dataset')
ax.bar_label(p1, padding=2)
ax.set_xticks(total_class)





#%
###############################################################################
###                  Wine quality data sets                          ##########
###############################################################################
### downloaded from here https://archive.ics.uci.edu/ml/datasets/Wine+Quality
### contains different ratings of wine using various measurement from each wine
### the output is 0 to 10 level scores
### 4898 different wine examples

data = pd.read_csv('data/winequality-white.csv')
val = data.values   # all values are saved as string in this dataset
numbers = []  # store the final values into numeric numbers

attri_names0 = data.keys()[0]
attri_names = attri_names0.replace('"','')
attibutes = attri_names.split(';')[0:-1] # names for all the attributes


# prepare all the data
for example in val: 
    one_obs = example[0]  # getting one observation, it was saved as string, and all attributes are together
    all_string = one_obs.split(';')
    all_val = [float(item) for item in all_string]
    numbers.append(all_val)

numbers = np.asarray(numbers)

# perpare the data and labels
wine_data  = numbers[:, 0:-1]
wine_labels = np.asarray(numbers[:, -1], dtype=int)

print('Total number of wine example selected:')
print(len(wine_data))



X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(wine_data, wine_labels, test_size=0.2, random_state=42)

X = StandardScaler().fit_transform(wine_data)
X_train_wine_standard, X_test_wine_standard, y_train_wine_standard, y_test_wine_standard = train_test_split(X, wine_labels, test_size=0.2, random_state=42)

# createa a list of numbers to training samples we want to use.
rates  = [0.3, 0.4, 0.6, 0.8,1]  # from 30% to 100% of all the training data
length = len(X_train_img)
lengths_img = [int(length*rate) for rate in rates]


length = len(X_train_wine)
lengths_wine = [int(length*rate) for rate in rates]



# investigate the number of classes in each case.
num_instances = []  # store the number of instances for each class
total_class = list(set(y_train_wine)) # figure out how many classes are there in the training data
y_train_wine_array = np.asarray(y_train_wine)
for a_class in total_class:
    num_instances.append( len(np.where(y_train_wine_array == a_class)[0]) )
    

# plot a figure to show the number of instances in each class in the training data
#plt.figure()
fig, ax = plt.subplots()
p1 = ax.bar(total_class, num_instances)
plt.ylabel('Number of instances')
plt.title('Data distribution across different classes in wine dataset')
ax.bar_label(p1, padding=2)
ax.set_xticks(total_class)



#%%


#% Define a function to conduct the experiment
# for differnet length of data used for training, conduct an experiment
def ConductExp(model, X_train_all, y_train_all, X_test_all, y_test_all, lengths):
    
    '''
    Inputs:
        
    model
        the input model to used to train in each experiment
    X_train_all,
        all the training data
    
    y_train_all,
        all the training labels
        
    X_test_all,
        all the testing data
        
    y_test_all,
        all the testing data
        
    lengths, 
        contains the lengths of the training data samples to use in each experiment
    
    '''

    train_accs = []
    test_accs  = []
    
    for n in range(len(lengths)):
        
        end = lengths[n]
        print('Conducting experiment', n+1, 'out of',  len(lengths))
        X_train = X_train_all[0:end]; y_train = y_train_all[0:end];
        model = model.fit(X_train, y_train)
    
        train_acc = model.score(X_train, y_train)
        pred_all  = model.predict(X_test_all)
        test_acc  = metrics.accuracy_score(y_test_all, pred_all)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
    return train_accs, test_accs




#%% Decision Tree

#############################################################
#####                   Decision Tree                   #####   
#############################################################

from sklearn import tree

'''
Decision Trees. 
For the decision tree, you should implement or steal a decision 
tree algorithm (and by "implement or steal" I mean "steal"). Be sure to use some 
form of pruning. You are not required to use information gain (for example, there 
is something called the GINI index that is sometimes used) to split attributes, 
but you should describe whatever it is that you do use.
'''


#############################################################
#####   Decision Tree experiments on MINIST image data
grid_param={"criterion":["gini","entropy"],
             "splitter":["best","random"],
             "max_depth":range(2,20,4),  # define early stopping based on depth, which is also pre-pruning of the tree
             "min_samples_leaf":range(1,10,2),  # The minimum number of samples required to be at a leaf node.
             "min_samples_split":range(2,6,2)   # The minimum number of samples required to split an internal node
            }

clf = tree.DecisionTreeClassifier()
grid_search=GridSearchCV(estimator=clf, param_grid=grid_param, 
                         scoring='accuracy', cv=5,n_jobs=-1, verbose=1)
grid_search.fit(X_train_img,y_train_img)


#% Use the best parameter to conduct training
best_para_img = grid_search.best_params_

print(best_para_img)

# initialize the model using the best hyperparameters
DT_model_img = tree.DecisionTreeClassifier(criterion=best_para_img['criterion'],
                                    max_depth        =best_para_img['max_depth'],
                                    min_samples_leaf =best_para_img['min_samples_leaf'],
                                    min_samples_split=best_para_img['min_samples_split'],
                                    splitter         =best_para_img['splitter'])


# conduct the experiment using different sizes of training data sets  
train_accs_DT_img, test_accs_DT_img = ConductExp(DT_model_img, X_train_img, y_train_img, X_test_img, y_test_img, lengths_img)



#%
# Figure is saving results for image data
plt.figure() 
plt.subplot(2,1,1)
plt.plot(lengths_img, train_accs_DT_img, marker='o', color='b')
plt.plot(lengths_img, test_accs_DT_img, marker='d',color='b')
plt.title('Decision tree experiments on different size of training data')
plt.ylabel('Accuracy/Img data')
plt.legend(['train', 'test', 'train_standard','test_standard'])


plt.figure()
plot_confusion_matrix(DT_model_img, X_test_img, y_test_img)  
plt.show()


#%%

#############################################################
#####   Decision Tree experiments on wine data  

clf = tree.DecisionTreeClassifier()
grid_search=GridSearchCV(estimator=clf, param_grid=grid_param, cv=5, 
                         scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_wine,y_train_wine)


#% Use the best parameter to conduct training
best_para_wine = grid_search.best_params_
print(best_para_wine)

#%%

# initialize the model using the best hyperparameters
DT_model_wine = tree.DecisionTreeClassifier(
                                    criterion        =best_para_wine['criterion'],
                                    max_depth        =best_para_wine['max_depth'],
                                    min_samples_leaf =best_para_wine['min_samples_leaf'],
                                    min_samples_split=best_para_wine['min_samples_split'],
                                    splitter         =best_para_wine['splitter'])


# conduct the experiment using different sizes of training data sets    
train_accs_DT_wine, test_accs_DT_wine = ConductExp(DT_model_wine, X_train_wine, y_train_wine, 
                                                   X_test_wine, y_test_wine, lengths_wine)


#%
# Figure is saving results for wine data
plt.subplot(2,1,2)
plt.plot(lengths_wine, train_accs_DT_wine, marker='o', color='b')
plt.plot(lengths_wine, test_accs_DT_wine, marker='d', color='b')

#plt.title('Decision tree experiments on Wine data')
plt.xlabel('Number of data samples used for training')
plt.ylabel('Accuracy/Wine data')
plt.legend(['train', 'test', 'train_standard','test_standard'])


plt.figure()
plot_confusion_matrix(DT_model_wine, X_test_wine, y_test_wine)  
plt.show()




#%%
# Neural networks

#############################################################
#####                      Neural networks              #####   
#############################################################

from sklearn.neural_network import MLPClassifier


#%
# Experiment on MINIST image data
# Neural network hyperparameter tuning on image data

clf = MLPClassifier(hidden_layer_sizes=(30,50,30),
                    solver='adam',random_state=1,
                    )

grid_param={"activation":["relu","tanh"],  # which activation function to use
             "learning_rate_init":[0.01, 0.001, 0.0001],  # define initial learning rate
             "shuffle":[True, False],      # whether to shuffle the data in each iteartion
             "alpha": [0.01, 0.001, 0.0001]  # L2 penalty (regularization term) parameter
            }
grid_search=GridSearchCV(estimator=clf, param_grid=grid_param, cv=5, 
                         scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_img,y_train_img)



#%
#% Use the best parameter to conduct training
best_para_img = grid_search.best_params_
print(best_para_img)

# initialize the mode using the best hyperparameters
MLP_model_img = MLPClassifier(hidden_layer_sizes=(30,50,30), solver = 'adam',random_state=1, verbose=False,
                            activation         = best_para_img['activation'],
                            alpha              = best_para_img['alpha'],    # L2 penalty (regularization term) parameter.
                            learning_rate_init = best_para_img['learning_rate_init'], # learning rate
                            shuffle            = best_para_img['shuffle'],  # whether to shuffle the data or not
                            )

# conduct the experiment using different sizes of training data sets    
train_accs_MLP_img, test_accs_MLP_img = ConductExp(MLP_model_img, X_train_img, y_train_img, 
                                           X_test_img, y_test_img, lengths_img)


#%
# Figure 1 is saving results for image data
plt.figure() 
plt.subplot(2,1,1)
plt.plot(lengths_img, train_accs_MLP_img, marker='o')
plt.plot(lengths_img, test_accs_MLP_img, marker='d')

plt.title('Neural networks experiments on different size of training data')
plt.ylabel('Accuracy/Img data')
plt.legend(['train', 'test'])

plt.figure()
plot_confusion_matrix(MLP_model_img, X_test_img, y_test_img)  
plt.show()
 
 
 
#%%

# Experiment on wine quality data
# Neural network hyperparameter tuning on image data

clf = MLPClassifier(hidden_layer_sizes=(30,50,30),solver='adam',random_state=1,)
grid_search=GridSearchCV(estimator=clf, param_grid=grid_param, cv=5, 
                         scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_wine,y_train_wine)


#%
best_para_wine = grid_search.best_params_
print(best_para_wine)

# initialize the mode using the best hyperparameters
MLP_model_wine = MLPClassifier(hidden_layer_sizes=(30,50,30), solver = 'adam',random_state=1, verbose=False,
                        activation         = best_para_img['activation'],
                        alpha              = best_para_img['alpha'],    # L2 penalty (regularization term) parameter.
                        learning_rate_init = best_para_img['learning_rate_init'], # learning rate
                        shuffle            = best_para_img['shuffle'],  # whether to shuffle the data or not
                        )

# conduct the experiment using different sizes of training data sets    
train_accs_MLP_wine, test_accs_MLP_wine = ConductExp(MLP_model_wine, X_train_wine, y_train_wine, 
                                           X_test_wine, y_test_wine, lengths_wine)

# Figure 1 is saving results for image data
#plt.figure() 
plt.subplot(2,1,2)
plt.plot(lengths_wine, train_accs_MLP_wine, marker='o')
plt.plot(lengths_wine, test_accs_MLP_wine, marker='d')


#plt.title('Decision tree experiments on Wine data')
plt.xlabel('Number of data samples used for training')
plt.ylabel('Accuracy/Wine data')
plt.legend(['train', 'test'])

plt.figure()
plot_confusion_matrix(MLP_model_wine, X_test_wine, y_test_wine)  
plt.show()




#%%

# Boosting
from sklearn.ensemble import GradientBoostingClassifier



grid_param={"n_estimators":  range(50, 200, 60),   # number of trees to created in the model
             "learning_rate":[0.1, 0.01],  # shrinks the contribution of each tree by learning rate
             "max_depth":    range(6,20,5),     # maximum number of iteration to run
             #"min_samples_split":[2, 4], # The minimum number of samples required to split an internal node
             #"min_samples_leaf": [2, 4]     # The minimum number of samples required to be at a leaf node.
            }


clf = GradientBoostingClassifier()
### Experiment on image dataset
grid_search=GridSearchCV(estimator=clf, param_grid=grid_param, cv=5, 
                         scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_img,y_train_img)

best_para_img = grid_search.best_params_
print(best_para_img)

# initialize the mode using the best hyperparameters
booster_model_img = GradientBoostingClassifier(
                       n_estimators       = best_para_img['n_estimators'],
                       learning_rate      = best_para_img['learning_rate'],    # L2 penalty (regularization term) parameter.
                       max_depth          = best_para_img['max_depth'], # learning rate
                       #min_samples_split  = best_para_img['min_samples_split'], 
                       #min_samples_leaf   = best_para_img['min_samples_leaf'],  # whether to shuffle the data or not
                       )


# conduct the experiment using different sizes of training data sets    
train_accs_boost_img, test_accs_boost_img = ConductExp(booster_model_img, X_train_img, y_train_img, 
                                           X_test_img, y_test_img, lengths_img)

#%
# Figure 1 is saving results for image data
plt.figure() 
plt.subplot(2,1,1)
plt.plot(lengths_img, train_accs_boost_img, marker='o')
plt.plot(lengths_img, test_accs_boost_img,  marker='d')

plt.title('Boosting experiments on different size of training data')
plt.ylabel('Accuracy/Img data')
plt.legend(['train', 'test'])

plt.figure()
plot_confusion_matrix(booster_model_img, X_test_img, y_test_img)  
plt.show()





#%
#### Experiment on wine data sets
clf = GradientBoostingClassifier()

grid_search=GridSearchCV(estimator=clf, param_grid=grid_param, cv=5, 
                         scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_wine,y_train_wine)

best_para_wine = grid_search.best_params_
print(best_para_wine)

# initialize the mode using the best hyperparameters
booster_model_wine = GradientBoostingClassifier(
                       n_estimators       = best_para_wine['n_estimators'],
                       learning_rate      = best_para_wine['learning_rate'],    # L2 penalty (regularization term) parameter.
                       max_depth          = best_para_wine['max_depth'], # learning rate
                       #min_samples_split  = best_para_wine['min_samples_split'], 
                       #min_samples_leaf   = best_para_wine['min_samples_leaf'],  # whether to shuffle the data or not
                       )

# conduct the experiment using different sizes of training data sets    
train_accs_boost_wine, test_accs_boost_wine = ConductExp(booster_model_wine, X_train_wine, y_train_wine, 
                                           X_test_wine, y_test_wine, lengths_wine)

# Figure 1 is saving results for image data
#plt.figure() 
plt.subplot(2,1,2)
plt.plot(lengths_wine, train_accs_boost_wine, marker='o')
plt.plot(lengths_wine, test_accs_boost_wine, marker='d')


#plt.title('Decision tree experiments on Wine data')
plt.xlabel('Number of data samples used for training')
plt.ylabel('Accuracy/Wine data')
plt.legend(['train', 'test'])

plt.figure()
plot_confusion_matrix(booster_model_wine, X_test_wine, y_test_wine)  
plt.show()



#%%
# Support Vector Machines


grid_param={"kernel": ['rbf', 'linear', 'sigmoid'], # transform the given dataset input data into required form.
            "C": [1,3,5],          # regularization parameter. The strength of the regularization is inversely proportional to C.
            "gamma": [0.1,0.3,0.5],  # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
           }


clf = svm.SVC(random_state=42)
### Experiment on image dataset
grid_search=GridSearchCV(estimator=clf, param_grid=grid_param, cv=5, 
                         scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_img,y_train_img)

best_para_img = grid_search.best_params_
print(best_para_img)

# # initialize the mode using the best hyperparameters
svm_model_img = svm.SVC(
                        kernel   = best_para_img['kernel'],
                        C        = best_para_img['C'],     # regularization parameter. The strength of the regularization is inversely proportional to C.
                        gamma    = best_para_img['gamma'], # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
                        #max_iter = best_para_img['max_iter'], # Hard limit on iterations within solver.
                        )


# conduct the experiment using different sizes of training data sets    
train_accs_svm_img, test_accs_svm_img = ConductExp(svm_model_img, X_train_img, y_train_img, 
                                           X_test_img, y_test_img, lengths_img)

#%
# Figure 1 is saving results for image data
plt.figure() 
plt.subplot(2,1,1)
plt.plot(lengths_img, train_accs_svm_img, marker='o')
plt.plot(lengths_img, test_accs_svm_img, marker='d')

plt.title('SVM experiments on different size of training data')
plt.ylabel('Accuracy/Img data')
plt.legend(['train', 'test'])

plt.figure()
plot_confusion_matrix(svm_model_img, X_test_img, y_test_img)  
plt.show()


# #%% plot learning curve
# from sklearn.model_selection import learning_curve

# svm_model_img = svm.SVC(
#                        kernel   = 'linear',
#                        C        = 1.0,     # regularization parameter. The strength of the regularization is inversely proportional to C.
#                        gamma    = 0.1, # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
#                        #max_iter = best_para_img['max_iter'], # Hard limit on iterations within solver.
#                        )


# train_sizes, train_scores, validation_scores = learning_curve(
#                             estimator = svm_model_img,
#                             X = X_train_img,
#                             y = y_train_img, train_sizes = lengths_img, cv = 5,
#                             scoring = 'neg_mean_squared_error')




#%
#### Experiment on wine data sets
clf = svm.SVC(random_state=42)
grid_search=GridSearchCV(estimator=clf, param_grid=grid_param, cv=5, 
                         scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_wine,y_train_wine)

best_para_wine = grid_search.best_params_
print(best_para_wine)

# initialize the mode using the best hyperparameters
svm_model_wine = svm.SVC(
                        kernel   = best_para_wine['kernel'],
                        C        = best_para_wine['C'],     # regularization parameter. The strength of the regularization is inversely proportional to C.
                        gamma    = best_para_wine['gamma'], # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
                        #max_iter = best_para_wine['max_iter'], # Hard limit on iterations within solver.
                        )


# conduct the experiment using different sizes of training data sets    
train_accs_svm_wine, test_accs_svm_wine = ConductExp(svm_model_wine, X_train_wine, y_train_wine, 
                                           X_test_wine, y_test_wine, lengths_wine)

# Figure 2 is saving results for wine data
plt.figure() 
plt.subplot(2,1,1)
plt.plot(lengths_wine, train_accs_svm_wine, marker='o')
plt.plot(lengths_wine, test_accs_svm_wine, marker='d')


#plt.title('Decision tree experiments on Wine data')
plt.xlabel('Number of data samples used for training')
plt.ylabel('Accuracy/Wine data')
plt.legend(['train', 'test'])





#%%
# k-nearest neighbors

from sklearn.neighbors import KNeighborsClassifier

#train_colors = ['#bdd7e7', '#6baed6', '#3182bd']
#test_colors  = ['#fdbe85', '#fd8d3c', '#e6550d']

train_colors = ['#2ca25f', '#e6550d', '#3182bd']
test_colors  = ['#2ca25f', '#e6550d', '#3182bd']


# initialize the mode using the best hyperparameters
neighbors = [3,6,10]
plt.figure() 
for n in range(len(neighbors)):
    
    n_neighbors = neighbors[n]
    
    knn_model_img = KNeighborsClassifier(n_neighbors=n_neighbors)
    # conduct the experiment using different sizes of training data sets    
    train_accs_knn_img, test_accs_knn_img = ConductExp(knn_model_img, X_train_img, y_train_img, 
                                               X_test_img, y_test_img, lengths_img)
    
    #%
    # Figure 1 is saving results for image data
    plt.subplot(2,1,1)
    plt.plot(lengths_img, train_accs_knn_img, marker='.', color=train_colors[n])
    plt.plot(lengths_img, test_accs_knn_img,  marker='d', color=test_colors[n])
    
    
    plt.ylabel('Accuracy/Img data')
    #plt.legend(['train', 'test'])
    
    
    knn_model_wine = KNeighborsClassifier(n_neighbors=n_neighbors)
    # conduct the experiment using different sizes of training data sets    
    train_accs_knn_wine, test_accs_knn_wine = ConductExp(knn_model_wine, X_train_wine, y_train_wine, 
                                               X_test_wine, y_test_wine, lengths_wine)
    
    # Figure 2 is saving results for wine data
    #plt.figure() 
    plt.subplot(2,1,2)
    plt.plot(lengths_wine, train_accs_knn_wine, marker='.', color=train_colors[n])
    plt.plot(lengths_wine, test_accs_knn_wine, marker='d',  color=test_colors[n])
    
    
    #plt.title('Decision tree experiments on Wine data')
    plt.xlabel('Number of data samples used for training')
    plt.ylabel('Accuracy/Wine data')
    #plt.legend(['train', 'test'])


# shrink current plot width
ax1=plt.subplot(2,1,1)
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.title('KNN experiments on different size of training data')
plt.legend(['train, k=3','test, k=3',
            'train, k=6','test, k=6', 
            'train, k=10','test, k=10', ], 
           loc='center left', bbox_to_anchor=(1, 0.5))

ax2=plt.subplot(2,1,2)
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(['train, k=3','test, k=3',
            'train, k=6','test, k=6', 
            'train, k=10','test, k=10', ], 
           loc='center left', bbox_to_anchor=(1, 0.5))


#%% plot all the algs figure together
# KNN use k=10, other algs are using the best hyperparameters that we get in each experiment

legends = ['DT','MLP','KNN', 'SVM', 'boost']

plt.figure()
plt.subplot(2,1,1)
plt.title('Performance across different algorithm on MINIST data')
plt.plot(lengths_img, train_accs_DT_img,  marker='.')
plt.plot(lengths_img, train_accs_MLP_img, marker='o')
plt.plot(lengths_img, train_accs_knn_img, marker='d')
plt.plot(lengths_img, train_accs_svm_img, marker='*')
plt.plot(lengths_img, train_accs_boost_img, marker='+')
plt.legend(legends)
plt.ylabel('Train accuracy')
plt.ylim([0.7, 1.1])


plt.subplot(2,1,2)
plt.plot(lengths_img, test_accs_DT_img,  marker='.')
plt.plot(lengths_img, test_accs_MLP_img, marker='o')
plt.plot(lengths_img, test_accs_knn_img, marker='d')
plt.plot(lengths_img, test_accs_svm_img, marker='*')
plt.plot(lengths_img, test_accs_boost_img, marker='+')
#plt.legend(legends)
plt.ylabel('Test accuracy')
plt.ylim([0.7, 1.1])
plt.xlabel('Number of data samples used for training')


plt.figure()
plt.subplot(2,1,1)
plt.title('Performance across different algorithm on wine data')
plt.plot(lengths_wine, train_accs_DT_wine, marker='.')
plt.plot(lengths_wine, train_accs_MLP_wine,marker='o')
plt.plot(lengths_wine, train_accs_knn_wine, marker='d')
plt.plot(lengths_wine, train_accs_svm_wine, marker='*')
plt.plot(lengths_wine, train_accs_boost_wine, marker='+')
plt.legend(legends)
plt.ylabel('Train accuracy')
#plt.ylim([0.2, 1])


plt.subplot(2,1,2)
plt.plot(lengths_wine, test_accs_DT_wine,  marker='.')
plt.plot(lengths_wine, test_accs_MLP_wine, marker='o')
plt.plot(lengths_wine, test_accs_knn_wine, marker='d')
plt.plot(lengths_wine, test_accs_svm_wine, marker='*')
plt.plot(lengths_wine, test_accs_boost_wine, marker='+')
#plt.legend(legends)
plt.ylabel('Test accuracy')
plt.xlabel('Number of data samples used for training')
#plt.ylim([0.2, 1])


#%% Futher explore KNN values

more_neighbors = range(1,30,3)
train_accs_more_knn_img = []
test_accs_more_knn_img  = []

# different KNN values in img dataset
X_train = X_train_img; y_train=y_train_img;
X_test_all = X_test_img; y_test_all = y_test_img; 

for n_neighbors in more_neighbors:
    print(n_neighbors)

    knn_model_img = KNeighborsClassifier(n_neighbors=n_neighbors)

    model = knn_model_img.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    pred_all  = model.predict(X_test_all)
    test_acc  = metrics.accuracy_score(y_test_all, pred_all)
    
    train_accs_more_knn_img.append(train_acc)
    test_accs_more_knn_img.append(test_acc)
    
    
plt.figure()
plt.plot(more_neighbors, train_accs_more_knn_img, '--o')    
plt.plot(more_neighbors, test_accs_more_knn_img,'--d')    
plt.title('Effect of different k values on MINIST data')
plt.xlabel('K values')    
plt.ylabel('Accuracy')  
plt.legend(['Training','Testing'])  
plt.ylim([0.6, 1.1])


#%
# different KNN values in wine dataset
more_neighbors = range(1,30,3)
train_accs_more_knn_wine = []
test_accs_more_knn_wine  = []

X_train = X_train_wine; y_train=y_train_wine;
X_test_all = X_test_wine; y_test_all = y_test_wine; 

for n_neighbors in more_neighbors:
    print(n_neighbors)

    knn_model_img = KNeighborsClassifier(n_neighbors=n_neighbors)

    model = knn_model_img.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    pred_all  = model.predict(X_test_all)
    test_acc  = metrics.accuracy_score(y_test_all, pred_all)
    
    train_accs_more_knn_wine.append(train_acc)
    test_accs_more_knn_wine.append(test_acc)
    
    
plt.figure()
plt.plot(more_neighbors, train_accs_more_knn_wine, '--o')    
plt.plot(more_neighbors, test_accs_more_knn_wine,'--d')    
plt.title('Effect of different k values on wine data')
plt.xlabel('K values')    
plt.ylabel('Accuracy')  
plt.legend(['Training','Testing'])  
plt.ylim([0.3, 1.1])