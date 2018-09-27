Most of these project files were originally cloned from this [Udacity github repository](https://github.com/udacity/RL-Quadcopter-2). 
Udacity's README file is duplicated [here](README_Udacity.md).

## Method

This project explores the concepts of Reinforcement Learning (RL). Deep RL is merely RL using deep neural networks.
These concepts are used to teach a (simulated) quadrotor drone how to successfully complete a flying task.

At the most basic level, RL is based on a view of the world as shown below.

![RL diagram](https://cdn-images-1.medium.com/max/1600/1*mPGk9WTNNvp3i4-9JFgD3w.png)

There is the agent (the software program) and the environment that it operates in. The agent is in state _S_ at time _t_, and takes an action _A_, which affects the environment. The environment in turn affects the agent, determining its new state _S_ at the next time step. In addition the agent receives a "reward" _R_, and then the process continues. The goal of RL is for the agent software to "learn" how to navigate the environment by choosing actions that maximize the reward it expects to receive.

In this project, the agent is software that controls the speed of a quadcopter's four rotors. The environment is a simulated world that determines forces and torques on the quadcopter, based on the rotor speeds, that change its pose (x,y,z,pitch,roll - collectively the state information, yaw isn't simulated since the drone is four-fold symmetric about the vertical axis). The goal for the student is to write an effective reward function by which the agent can learn to achieve a flight task, in this case, take off and reach a hover altitude.

There are multiple RL algorithms depending on the agent design. Two different types of RL algorithms were explored in this project: Deep Q-Learning (DQL) and [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971). The former discretizes the action space (possible actions are discrete and countable), whereas the latter allows for continuous action variables (e.g., continuous rotor speeds) and employs a so-called "actor-critic method".

Notes | File | Usage
-- |  --- | ---
[1] | Quadcopter_Project_DDPG.ipynb | main notebook for the DDPG implementation, where the student implements the training and plotting functions
[1] | Quadcopter_Project_DQL.ipynb | main notebook for the DQL implementation 
[1] | task_AC.py |  All task files implement agent support functions init, step, reset and reward. The step function also steps the physics simulation. This file implements the agent task using the Actor-Critic (DDPG) method, independent rotor freedom.
[1] | task_AC_nopitchandroll.py | same as above, but locking all rotors to the same speed 
[1] | task_DQL_withpitchandroll.py |   implementation of the agent task using the DQL method, independent rotor freedom 
[1] | task_DQL_nopitchroll.py | same as above, but locking all rotors to the same speed 
[2] | physics_sim.py | code implementing the environment including drone flight characteristics and laws of physics. Normally not touched by the student, but it contained a bug I had to fix, and in order to do what I wanted I had to enhance it's capability 
NA | agents | folder containing agent implementations. Each agent implements init, reset, act, step, and learn functions
[1] | agents/DDPGagent.py | implementation of the agent in the DDPG method 
[1] | agents/DQLagent.py | implementation of the agent in the DQL method 
[1] | agents/DQLnetwork.py | implementation of the neural network for the DQLagent using Keras
[2] | agents/OUNoise.py | implementation of Ornstein-Uhlenbeck noise for the DDPG method 
[1] | agents/actor_orig.py | implementation of the actor NN in the DDPG method with 32x64x32 hidden layers using Keras
[1] | agents/actor_128x256x128.py | implementation of the actor NN in the DDPG method with 128x256x128 hidden layers using Keras
[1] | agents/critic_orig.py | implementation of the critic dual NNs in the DDPG method with 32x64 hidden layers using Keras
[1] | agents/critic_128x256.py | implementation of the critic dual NNs in the DDPG method with 128x256 hidden layers using Keras
[2] | agents/replaybuffer.py | used for batch training in both methods

- [1] Implemented by the student using supplied sample/skeleton code
- [2] Supplied

## Results

