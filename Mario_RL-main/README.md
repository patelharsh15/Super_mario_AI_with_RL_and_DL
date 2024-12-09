# Building a AI-Powered Mario using Double Deep Q Learning

## Environment Setup

Using conda environments to install:

For GPU supported torch: `conda env create -f mario_gpu.yml`

For CPU supported torch: `conda env create -f mario_cpu.yml`

# Learning the Environment

Mario learns by running: `python main.py`

The main function strips down the environment into its barebones to have a easier mario to learn.

This preprocessing includes:
- Skipping frames
- Turning the game environment into gray-scale
- Resizing the environment
- Transforming the environment
- Stacking multiple frames of the environment on top of one another

It then has Mario act based on the <strong>Double Q-Learning algorithm</strong>. I use a convolutional network with 3 convolutional networks and 2 fully-connected ones.

The Mario agent trains on 45,000 total episodes, which each episode representing the Mario reaching the end of the level or reaching a maximum number of iterations.

## Evaluation of results:
The first iteration Mario agent is unable to get past the 3rd green pipe.

Halfway through the episodes trained, the Mario agent is able to get past the third green pipe, but is inconsistently able to make the jumps.

After training for the full 45,000 episodes, the Mario agent is consistently able to get past the green pipe and make the two jumps, is sometimes able to make the jump past the two pairs of Goomba's (the brown creatures), and is inconsistently able to finish the game.

## Time to train

It took 86 hours on a NVIDIA 1660 Super to train. The exceedingly long time is a result of poor performance of my GPU, and the fact that I could only store 10,000 experiences of the agent at a time. Having more memory in the GPU would definitely improve the time performance of training.
