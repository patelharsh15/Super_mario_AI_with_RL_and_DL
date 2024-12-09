import os
import time
import wandb
from pathlib import Path
import datetime

from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from wrappers import SkipFrame, ResizeObservation
from agent import Mario


def main():
    wandb.init(project="Mario RL",
               config={
                   "batch_size": 32,
                   "architecture": "DDQN",
                   "dataset": "mario_gymnasium",
                   "episodes": 40000,
               }
    )


    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env,
                      [['right'],
                      ['right', 'A']]
                     )

    # Apply preprocessing
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)

    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    if not os.path.exists(save_dir):
        save_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
    episodes = 45000
    for e in range(episodes):
        state = env.reset()
        while True:
            action = mario.act(state)
            next_state, reward, done, info = env.step(action)
            mario.cache(state, next_state, action, reward, done)
            q, loss = mario.learn()
            state = next_state
            if done or info['flag_get']:
                break
        if q and loss and e % 20 == 0:
            wandb.log({
                'td-est-mean': q, 
                'loss': loss, 
                'epsilon': mario.epsilon,
                'episode': e,
                'step': mario.step
                })
    end = time.time()
    print("Time elapsed: ", end - start, "seconds")

if __name__ == '__main__':
    main()


