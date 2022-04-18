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



#% Frozen lake example

'''
Using module OpenAI's Gym
https://gym.openai.com/
'''

import numpy as np
import gym
import random, time

#%
env = gym.make('FrozenLake8x8-v1')

env.action_space       # number of actions
env.observation_space  # number of action_space
env.env.P[0][1] # transicion probability from state 0 to action 1.


#% Value iteration and Policy Iteration Helpter Functions

'''


Useful link to refer for policy iteration:
    https://medium.com/swlh/frozen-lake-as-a-markov-decision-process-1692815ecfd1

Useful link to refer to value iteartion:
    https://towardsdatascience.com/value-iteration-to-solve-openai-gyms-frozenlake-6c5e7bf0a64d
    https://medium.com/analytics-vidhya/solving-the-frozenlake-environment-from-openai-gym-using-value-iteration-5a078dffe438    

Policy evaluation:
    Determine the state-value function Vpi(s), for a given policy(pi).
    
Policy iteration is iterating the update rule until the change in Value estimate over
iteration becomes negligible.     


Policy control, improving the existing policy(pi)
In this case, acting greedy on expected value function which give us deterministic policy.
Taking an action that has the highest value from the states.

'''

class RF_Learning():
    
    def __init__(self, env, gamma=0.99, theta=0.1, iter_type='value_iter'):

        '''
        initialize the value function(V) and policy(pi) arbitrarily.
        
        '''
        
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.iter_type = iter_type
        
        num_action = env.action_space.n  # the number of actions the agent can take
        num_states = env.observation_space.n    # the number of states in the environment
        # random deterministic policy
        pi_init = []
        for s in range(num_states):
             p_s = [0] * num_action
             random_action = random.randint(0, num_action-1)
             p_s[random_action] = 1
             pi_init.append(p_s)
         
        # initialize state values
        self.V  = [0] * num_states  # assign the initial value function with all 0s.
        self.pi_init = pi_init
        
        self.policy = pi_init # for policy iteration

        
    
    def getAction(self, s):
        
        '''
        return an action for a given state.    
        s,  state
            
        '''
        
        # for the first time,,
        # using initialization to get the random action for a state
        current_action_from_init = np.asarray( self.policy[s] ) # at each state, the related action is assigned to 1, other actions are 0
        action = np.where(current_action_from_init==1)[0][0]
        
        return action

    
    def myargmax(self, V, pi, s, vis=False):
        '''
        
        POLICY CONTROL: IMPROVING THE EXISTING POLICY(pi)
        Act greedy on the expected value function which gives us the 
        deterministic policy.
        
        Given a state(s) and state values, generate a policy so that it follows the 
        maximum state values for its succesor states.
        
        Inputs:
        env, 
            a "gym" environment variable we are trying to solve.
        V,  
            the state values  
        pi, 
            the initialized policy with size (num_states, num_actions)    
        s, 
            which state is in right now.
        gamma, 
            discount factor
            
        Outputs:
        pi,
            the updated policy
        '''
        env   = self.env
        gamma = self.gamma
        num_action    = self.env.action_space.n  # the number of actions the agent can take
        action_values = np.zeros(num_action)  # define an array with the length of number of action.
    
        # itearate over each action
        for a in range(num_action):
            q = 0
            # for a given state and action, what is the transition probability
            '''
            Example of the transition probability given a state and action:
                env.env.P[0][1],  transicion probability from state 0 to action 1.
                    [(0.3333333333333333, 0, 0.0, False),
                     (0.3333333333333333, 8, 0.0, False),
                     (0.3333333333333333, 1, 0.0, False)]
                It has 33% chance of stay at state (0), and the reward of that action is 0, 
                it is not the goal or ending cell of the grid world.  
            '''
            # given a state and an action, get its transition probability, see example above.
            P = np.array(env.env.P[s][a])
            
            # num_s_, 
            #    the number of possible successor states can ended up, 
            #    given the current state and action
            # y, the number of elements in the transition probability matrix given this state and action
            #       from the example above, y is 4:
            #       probability, next state, reward, True or False for ending state
            (num_s_,y) = P.shape
            
            # iterate through each successor state
            for i in range(num_s_):
                # what is the transition probability, successor state and reward
                # for this successor state
                (p, s_, r, _) = P[i]  # i.e.  (0.3333333333333333, 8, 0.0, False)
                s_ = int(s_)
                q = q + p*(r + gamma*V[s_])  # calculate the current action_ value
                action_values[a] = q
    
        max_a = np.argmax(action_values)  # find the action that can results the largest action value
        
        pi[s][max_a] = 1
        
        if vis:
            print('state:', s)
            print('action: ', max_a)
            
        return pi



    

    
    def optimality_update(self, s,
                          policy_action=None ):
        '''
    
        This function is updating the state_value V[s] by taking action which maximizes current value.
        
        
        inputs:
        env,
             a "gym" environment variable we are trying to solve.
        V,  
            the state values, the same length as the number of states    
        s, 
            which state is in right now.
        gamma, 
            discount factor
            
        iter_type,
            'value_iter' or 'policy_iter'
            
         
        for policy iteration,
           policy_action, 
               the action user wants to use within this policy. 
               Optional, default None.  
           pi_init,
               the initialized policy. 
               During policy iteration, each time when given a state to update the state value
               need to draw a initializad action given that state from this pi_init.
           
        
        '''
        env        = self.env
        gamma      = self.gamma
        iter_type  = self.iter_type
        num_action = self.env.action_space.n         # the number of actions the agent can take
        # num_states = env.observation_space.n    # the number of states in the environment
        
        #pi = np.zeros((num_states, num_action))
        action_values = np.zeros(num_action)    # define an array with the length of number of action.
    
     
        if iter_type == 'value_iter':
            
            ######################################
            ##     Used for Value Iteration     ##
            ######################################
            
            '''
            In value iteration, we preform just one sweep over all the states and act greedily with current
            value function, instead of running policy evaluation at each iteartion. 
            
            ''' 
            
           # Step 1: find action which gives maximum value        
        
            # iterate all possible actions
            for a in range(num_action):
       
                q = 0 # initial q value
            
                P = np.array(env.env.P[s][a]) # transition probability given a state and action
                
                
                (num_s_,y) = P.shape # number of succesor states, and its attributes in each state (refer my argmax() above for more info)
                
                # iterate through each successor state
                for i in range(num_s_):
                    # what is the transition probability, successor state and reward
                    # for this successor state
                    (p, s_, r, done) = P[i]  
                    
                    s_ = int(s_)
                    q = q + p*(r + gamma*self.V[s_]) # calculate the current action_ value
                
                action_values[a] = q
                
                
                max_a = np.argmax(action_values)    # find the action that can results the largest action value
                self.V[s] = action_values[max_a]    # update my state value V[s]
            
            
        elif iter_type == 'policy_iter':

                ######################################
                ##     Used for Policy Iteration    ##
                ######################################

                # if user didn't assign an action, get a random action
                #   from the initialization of the state-action
                if policy_action is None:
                    policy_action = self.getAction(s)
                
               
                u = 0 # initial q value
                P = np.array(env.env.P[s][policy_action]) # transition probability given a state and action
                
                (num_s_,y) = P.shape # number of succesor states, and its attributes in each state (refer my argmax() above for more info)
                
                # iterate through each successor state
                for i in range(num_s_):
                    # what is the transition probability, successor state and reward
                    # for this successor state
                    (p, s_, r, done) = P[i]  
                    
                    s_ = int(s_)
                    # for policy iteration, here we get the expectation of the 
                    # given state, instead of the argmax in value iteration
                    u = u + p*(r + gamma*self.V[s_]) # calculate the current action_ value
                
                self.V[s] = u   # update my state value V[s]
     

    
        
    def value_iteration(self, ):
    
        '''
        
        inputs:
        env,
             a "gym" environment variable we are trying to solve.
        gamma, 
            discount factor
            
        theta,
            the change in value estimate becomes negligible (<theta), the given policy
            will stricly converge into Optimal policy.
        
        '''
        t1 = time.time()
        
        gamma = self.gamma
        theta = self.theta
        num_action = self.env.action_space.n         # the number of actions the agent can take
        num_states = self.env.observation_space.n    # the number of states in the environment
        
        #V = np.zeros(num_states)
        
        
        iter_num = 0  # track number of iterations
        while True:
            delta = 0
            iter_num+=1
            # iterate through all states
            for s in range(num_states):
                v = self.V[s]           
                self.optimality_update(s, gamma)  # update my state values           
                delta = max(delta, abs(v - self.V[s]))  # the change between the old and new state value
                
            if delta < theta: break  # stop when the change is smaller than a pre defined "theta"
            
        pi = np.zeros((num_states, num_action))
        # after the value iteration is done, get the policy
        for s in range(num_states):
             pi = self.myargmax(self.V, pi, s)
               
        # return optimal value function and optimal policy
        t2 = time.time()
        self.iter_num = iter_num  # number of iteration to reach converge
        #self.pi = pi              # the optimal policy we found
        self.time_cost = t2-t1    # time needed to converge
        self.policy = pi 
    
    
    
        
    def evaluate_policy(self):
    
        '''
        
        This function is similar to policy_evaluation(), the main difference is how
        to call optimality_update() to update the state value.
        
        inputs:
        env,
             a "gym" environment variable we are trying to solve.
        gamma, 
            discount factor
            
        theta,
            the change in value estimate becomes negligible (<theta), the given policy
            will stricly converge into Optimal policy.
            
        pi_init, 
            initialized policy    
        
        '''
        
        theta = self.theta
        #num_action = self.env.action_space.n         # the number of actions the agent can take
        num_states = self.env.observation_space.n    # the number of states in the environment

        
        #V = np.zeros(num_states)
        #iter_num = 0  # track number of iterations
        while True:
            delta = 0
            #iter_num+=1
            # iterate through all states
            for s in range(num_states):
                v = self.V[s]  
                # evaluate policy for a state
                self.optimality_update(s)  # update my state values           
                delta = max(delta, abs(v - self.V[s]))  # the change between the old and new state value
                
            if delta < theta: break  # stop when the change is smaller than a pre defined "theta"
            
    

    
    def improve_policy_for_state(self, s):
        
        '''
        Given a state (s), return an action that improves the existing policy
    
        '''
        num_action = self.env.action_space.n   
        
        
        v_s_max = float("-inf")
        for action in range(num_action):
            self.optimality_update(s, policy_action=action)  # update my state value V[s]          
    
            if self.V[s] > v_s_max:
                v_s_max = self.V[s]  # update v_s_max 
                action_max = action
        return action_max




    def improve_policy(self,vis=False):
        '''
        Use the current value function to improve the current state
        '''
        num_states = self.env.observation_space.n 
        
        policy_stable = True
        #vis and print(self.policy)
        
        for s in range(num_states):
            current_action = self.getAction(s)
            # act greedy to use the action that cause the state has the largest state value
            action_max     = self.improve_policy_for_state(s)
            vis and print(f"State {s}, Action {current_action}, New action {action_max}")
            if action_max != current_action:
                policy_stable = False
                self.policy[s][current_action] = 0
                self.policy[s][action_max] = 1
        
        return policy_stable
        
    
    
    

    def policy_iteration(self, vis=False):
        
        
        t1 = time.time()
        policy_stable = False
        iter_num = 0  # track number of iterations
        
        while (not policy_stable):
            iter_num +=1 # track number of iterations
            self.evaluate_policy()
            policy_stable = self.improve_policy() # if policy is stable, then it's the optimal policy
        
                # return optimal value function and optimal policy
        t2 = time.time()
        # the optimal policy we found
        self.time_cost = t2-t1    # time needed to converge
        self.iter_num = iter_num 




    def myQLearning(self, alpha=0.628, max_iter=200, episode=1000, vis=False):
        '''
        Construct a Q-table to host state-action-rewards value pairs.
        
        max_iter,
            max number of iteration for  each episode. If not done yet, force it to stop.
        
        '''

        reward_list = [] # rewards per episode calculate
        
        num_action = self.env.action_space.n         # the number of actions the agent can take
        num_states = self.env.observation_space.n    # the number of states in the environment
        
        Q = np.zeros([num_states,num_action])
        iter_total = 0
        
        
        t1 = time.time()
        for i in range(episode):
            if i%500 == 0 and vis: print('Working on episode: {}/{}'.format(i,episode))
            s = env.reset()   # reset the environment to get a initialized state
            done = False      # whether it is ending state or not
            iter_tracker = 0
            total_reward = 0
            # set a max iteration.
            #    if by max iteration, the agent has not done (either in the hole or reach target state)
            #    then manually stop this process, and reset the environment to search again.
            
            while iter_tracker< max_iter:
                
                iter_tracker +=1  # track the number of iterations that has been ran
               
                #The Q-Table learning algorithm
                # Choose action from Q table
                # One version of epsilon greedy implementation.
                #     as number of episode increases, the random initiated value will become smaller and smaller
                #     meaning will trust the Q value more and more.
                a = np.argmax(Q[s,:] + np.random.randn(1,num_action)*(1./(i+1)) )
                # Get new state & reward from environment
                s_, r, done,_ = env.step(a)  # sucessor state, reward, ending state, probability
                # Update Q-Table with new knowledge
                Q[s,a] = Q[s,a] + alpha*(r + self.gamma*np.max(Q[s_,:]) - Q[s,a])
                total_reward += r
                s = s_
                
                if done == True: break
   
            iter_total +=  iter_tracker       # total iteartions updating Q table across all episodes
            reward_list.append(total_reward)  # total reward list for all episodes
        
        t2 = time.time()    
        self.iter_total=iter_total   
        self.Q=Q
        self.reward_list=reward_list
        self.episode = episode
        self.time_cost = t2-t1    # time to finish the solution



#%% value iteartion

#################################################
####         Value iteration experiment       ### 
#################################################

env = gym.make('FrozenLake8x8-v1')
gamma = 0.99
theta = 0.0001

myAgent = RF_Learning(env, gamma, theta, iter_type='value_iter')
myAgent.value_iteration()

    
print('when Theta is: ', theta)
print(myAgent.V)
print('Total iteration is: ', myAgent.iter_num)

    
# prepare optimal policyc
optimal_val_iter = np.argmax(myAgent.policy, axis=1)
print(optimal_val_iter.reshape((8,8)))

print('Time to converge: ', myAgent.time_cost)


#%%
# run experiment 500 episodes    
succ = 0  # track number of successful trial
exp_episode =  500  
t1 = time.time()
for i in range(exp_episode):

    s = env.reset()

    done = False
    while not done:
        s_, reward, done, info = env.step(optimal_val_iter[s])
        s=s_
        if done:
            if reward == 1:
                succ+=1
                break

t2=time.time()
print("Agent succeeded to reach goal {} out of 500 episodes using this policy".format(succ))
print('Experiment time cost:', t2-t1)    


    
    
#%% policy iteartion

#################################################
####         Policy iteration experiment      ### 
#################################################

gamma = 0.99
theta = 0.0001

myAgent = RF_Learning(env, gamma, theta, iter_type='policy_iter')
myAgent.policy_iteration(vis=False)

print('when Theta is: ', theta)
print(myAgent.V)
print('Total iteration is: ', myAgent.iter_num)

# prepare optimal policy
optimal_policy_iter = np.argmax(myAgent.policy, axis=1)
print(optimal_policy_iter.reshape((8,8)))

print('Time to converge: ', myAgent.time_cost)

#%%

# run experiment 500 episodes    
succ = 0  # track number of successful trial
exp_episode =  500  
t1 = time.time()
for i in range(exp_episode):

    s = env.reset()
    done = False
    
    while not done:
        s_, reward, done, info = env.step(optimal_policy_iter[s])
        s=s_
        if done:
            if reward == 1:
                succ+=1
                break

t2=time.time()
print("Agent succeeded to reach goal {} out of 500 episodes using this policy".format(succ))
print('Experiment time cost:', t2-t1)    

#%%

import matplotlib.pyplot as plt
value_iters = []
policy_iters = []

gammas = [0.6, 0.65, 0.7, 0.75, 0.79, 0.89, 0.95, 0.99]
for gamma in gammas:
    print('Gamma is ', gamma)
    myAgent_policy_iter = RF_Learning(env, gamma, theta, iter_type='policy_iter')
    myAgent_policy_iter.policy_iteration(vis=False)
    policy_iters.append(myAgent_policy_iter.iter_num)
    
    myAgent_value_iter = RF_Learning(env, gamma, theta, iter_type='value_iter')
    myAgent_value_iter.value_iteration()

    value_iters.append(myAgent_value_iter.iter_num)
    

plt.figure()
plt.plot( gammas, policy_iters, 'o--')    
plt.plot( gammas, value_iters, 'o--')    
plt.legend(['policy iteration','value iteartion'])
plt.ylabel('Number of iterations')
plt.xlabel('gamma value')

#%% Q-learning


myAgent = RF_Learning(env)
myAgent.myQLearning(alpha=0.628, episode=100000, vis=False)


print("Final Values Q-Table")
QTable = myAgent.Q
print(QTable)
print("Reward Sum on all episodes " + str(sum(myAgent.reward_list)/myAgent.episode))
print('Total iteration is: ', myAgent.iter_total)
print('Time to finish updating Q table: ', myAgent.time_cost)


#%%
# run experiment 500 episodes  
exp_episode =  500  
succ = 0  # track number of successful trial

t1 = time.time()
for i in range(exp_episode):
    
    s = env.reset()
    done = False
    
    while not done:
         action = np.argmax(QTable[s])
         s_, reward, done, info = env.step(action)
         s = s_

         if done:
            if reward == 1:
                succ+=1
                break

t2=time.time()
print("Agent succeeded to reach goal {} out of 500 episodes using this policy".format(succ))
print('Experiment time cost:', t2-t1)    





