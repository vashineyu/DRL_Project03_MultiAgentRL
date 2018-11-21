from unityagents import UnityEnvironment
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_episodes', default = 2000, type = int)
parser.add_argument('--experiment_tag', default = None, type = str)
parser.add_argument('--use_noise', default = 1)
FLAGS = parser.parse_args()

env = UnityEnvironment(file_name="./Tennis_Linux_NoVis/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

from maddpg import MADDPG
from collections import deque
import torch
agent = MADDPG(24, 2, 0)

env_info = env.reset(train_mode=True)[brain_name]
env_info.vector_observations.shape

def maddpg(max_episodes=2000, print_every=10):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, max_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = 0
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            #print(rewards) # example: [-0.009999999776482582, 0.0]
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += np.max(rewards)
            if any(dones):
                break 

        scores_deque.append(score)
        scores.append(score)
        torch.save(agent.actor_local.state_dict(), './actor.pth')
        torch.save(agent.critic_local.state_dict(), './critic.pth')
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, scores[-1], np.mean(scores_deque)))

        if len(scores_deque) == 100 and np.mean(scores_deque) >= 1.0:
            if i_episode % print_every != 0:
                print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, scores[-1], np.mean(scores_deque)))
            print('Environment solved!')
            break

    return scores

scores = maddpg(max_episodes = FLAGS.num_episodes)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure(figsize = (6,4))
plt.plot(np.arange(1, len(scores)+1), scores)
plt.title("Result")
plt.xlabel('Episode', fontsize = 16)
plt.ylabel('Average Scores', fontsize = 16)
plt.tight_layout()
if FLAGS.experiment_tag is not None:
    sav_name = "result_" + FLAGS.experiment_tag + ".png"
else:
    sav_name = "result.png"
plt.savefig(sav_name)
