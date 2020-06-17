# Project 1: Navigation using DQN

### Introduction

The goal of this project is to train an agent to navigate (and collect bananas!) in a large, square world, using a Deep Q-Network (DQN).  

![Sample Navigation](img/sample_navigation.gif "Sample Navigation")

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

This project's environment is based on [Unity ML-Agents environments](https://github.com/Unity-Technologies/ml-agents). Do the following steps to setup the environment:

1. Download the Unity environment from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the project folder and unzip it.

3. To install required dependencies, follow [these steps](https://github.com/udacity/deep-reinforcement-learning#dependencies).

    I had to also update PyTorch version, as I had some issues locally when calling `.to(device)` on PyTorch tensors. Also, as I am training in a computer with an NVIDIA GPU (RTX 2070), I installed CUDA Toolkit. I also installed [PyTorch Summary](https://github.com/sksq96/pytorch-summary) to visualize the model:
    ```
    > conda install pytorch cudatoolkit=10.2 -c pytorch
    > pip install torchsummary
    ```


### Instructions

All the training procedures are inside `Navigation.ipynb`. Run all the cells to train the agent, or just run the imports cell and the Section 5, which loads trained weights into a new agent.

There are also two additional files:
- `model.py`: defines the neural network architecture using PyTorch;
- `dqn_agent.py`: implements DQN using the model above.
