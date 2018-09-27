Most of these project files were originally cloned from this [Udacity github repository](https://github.com/udacity/RL-Quadcopter-2). 
Udacity's README file is duplicated [here](README_Udacity.md).

## Method

This project explores the concepts of Reinforcement Learning (RL). Deep RL is merely RL using deep neural networks.
These concepts are used to teach a (simulated) quadrotor drone how to successfully complete a flying task.

At the most basic level, RL is based on a view of the world as shown below.

![RL diagram](https://cdn-images-1.medium.com/max/1600/1*mPGk9WTNNvp3i4-9JFgD3w.png)

There is the agent (the software program) and the environment that it operates in. The agent is in state _S_ at time _t_, and takes an action _A_, which affects the environment. The environment in turn affects the agent, determining its new state _S_ at the next time step. In addition the agent receives a "reward" _R_, and then the process continues. The goal of RL is for the agent software to "learn" how to navigate the environment by choosing actions that maximize the reward it expects to receive.

In this project, the agent is software that controls the speed of a quadcopter's four rotors. The environment is a simulated world that determines forces and torques on the quadcopter, based on the rotor speeds, that change its pose (x,y,z,pitch,roll,yaw - collectively the state information). The goal for the student is to write an effective reward function by which the agent can learn to achieve a flight task, in this case, take off and reach a hover altitude.

There are multiple RL algorithms depending on the agent design. Two different types of RL algorithms were explored in this project: Deep Q-Learning (DQL) and [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971). The former discretizes the action space (possible actions are discrete and countable), whereas the latter allows for continuous action variables (e.g., continuous rotor speeds) and employs a so-called "actor-critic method".

## Results


