#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 22:21:52 2022

@author: mingmingzhang

Assiggment4

Some useful links to review MDPs or MRPs
https://towardsdatascience.com/introduction-to-reinforcement-learning-markov-decision-process-44c533ebf8da

very useful studying materials
https://towardsdatascience.com/reinforcement-learning-markov-decision-process-part-2-96837c936ec3
"""


import mdptoolbox.example
import matplotlib.pyplot as plt
import numpy as np

'''
Refer to this page for more info.

https://pymdptoolbox.readthedocs.io/en/latest/api/example.html

A forest is managed by two actions: ‘Wait’ and ‘Cut’. 
An action is decided each year with first the objective to maintain an old forest 
for wildlife and second to make money selling cut wood.

Each year there is a probability p that a fire burns the forest.

Here is how the problem is modelled. Let {0, 1 . . . S-1 } be the states of the forest, 
with S-1 being the oldest. Let ‘Wait’ be action 0 and ‘Cut’ be action 1.
After a fire, the forest is in the youngest state, that is state 0. 

'''

# CREATE A FOREST MANAGEMENT PROBLEM
# P, transition probability (A x S x S)
# R, reward matrix (S X A)
firerate=0.05   # probability of having a fire
discount=0.8    # discount rate for the future value
                # 0 means more importance is given to immediate rewards.
                # 1 means more importance is given to future rewards.
P, R = mdptoolbox.example.forest(S=800,  # The number of states, an integer greater than 1.
                                 r1=5, # the reward when forest is in its oldest state and action 'Wait' to be performed
                                 r2=50, # the reward when forest is in its oldest state and action 'Cut' to be performed
                                 p=firerate, # the probability of wild fire occurence, in range(0,1). 
                                 is_sparse=False)



def plot_reward_action(vi):
    plt.figure()
    plt.plot(np.arange(0, len(vi.policy)), vi.policy, '.--')
    plt.xlabel('Number of states')
    plt.ylabel('Action. 0: wait; 1: cut.' )
    plt.ylim([-0.5,2])
    
    plt.figure()
    plt.bar(np.arange(0, len(vi.V)), vi.V)
    plt.title('Expected reward as a function of states')
    plt.xlabel('Number of states')
    plt.ylabel('Expected reward')

#%%
# solving the problem using value iteartion
print('This is my value iteartions \n')
vi = mdptoolbox.mdp.ValueIteration(P, R, discount=discount)
vi.run()
print('Discount rate: ', vi.discount)
print('fire rate:', firerate)
print('Iterations until converge: ', vi.iter,'\n')
print('Optimal policy: ', vi.policy, '\n')
print('Expected Reward in each state: ', vi.V,'\n')
print('Elapse time: ', vi.time)


    
plot_reward_action(vi)    
#%%
# solving the problem using policy iteartion
print('This is my policy iteartions \n')
vi = mdptoolbox.mdp.PolicyIteration(P, R, discount=discount)
vi.run()
print('Discount rate: ', vi.discount)
print('fire rate:', firerate)
print('Iterations until converge: ', vi.iter,'\n')
print('Optimal policy: ', vi.policy, '\n')
print('Expected Reward in each state: ', vi.V,'\n')
print('Elapse time: ', vi.time)


plot_reward_action(vi)    


#%%

# solving the problem using Q learning
print('This is my Q-learning \n')
vi=mdptoolbox.mdp.QLearning(P, R, discount=discount, n_iter=30000000)
vi.run()
print('Discount rate: ', vi.discount)
print('fire rate:', firerate)
print('Iterations until converge: ', vi.max_iter,'\n')
print('Optimal policy: ', vi.policy, '\n')
print('Expected Reward in each state: ', vi.V,'\n')
print('Elapse time: ', vi.time)

plot_reward_action(vi)  

#%%

plt.figure()
plt.plot(np.arange(0, len(vi.policy)), vi.policy, '.--')
plt.xlabel('Number of states')
plt.ylabel('Action. 0: wait; 1: cut.' )
plt.ylim([-0.5,2])

plt.figure()
plt.bar(np.arange(0, len(vi.V)), vi.V)
plt.title('Expected reward as a function of states')
plt.xlabel('Number of states')
plt.ylabel('Expected reward')



#%%


P, R = mdptoolbox.example.rand(S=800, A=10, is_sparse=False, mask=None)    

