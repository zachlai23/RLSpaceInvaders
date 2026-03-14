# This file is taken from MinAtar github : https://github.com/kenjyoung/MinAtar

import random, numpy, argparse
from minatar import Environment

NUM_EPISODES = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--game", "-g", type=str)
args = parser.parse_args()

env = Environment(args.game)

e = 0
returns = []
frame_counts = []
num_actions = env.num_actions()

# Run NUM_EPISODES episodes and log all returns
while e < NUM_EPISODES:
    # Initialize the return for every episode
    G = 0.0
    frames = 0

    # Initialize the environment
    env.reset()
    terminated = False

    #Obtain first state, unused by random agent, but inluded for illustration
    s = env.state()
    while(not terminated):
        # Select an action uniformly at random
        action = random.randrange(num_actions)

        # Act according to the action and observe the transition and reward
        reward, terminated = env.act(action)

        # Obtain s_prime, unused by random agent, but inluded for illustration
        s_prime = env.state()

        G += reward
        frames += 1

    # Increment the episodes
    e += 1

    # Store the return and frame count for each episode
    returns.append(G)
    frame_counts.append(frames)

print("Avg Return: " + str(numpy.mean(returns))+"+/-"+str(numpy.std(returns)/numpy.sqrt(NUM_EPISODES)))
print("Avg Frames Survived: " + str(numpy.mean(frame_counts))+"+/-"+str(numpy.std(frame_counts)/numpy.sqrt(NUM_EPISODES)))

