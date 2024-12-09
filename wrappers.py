import gym
import cv2
from gym.spaces import Box
import numpy as np


class ResizeObservation(gym.ObservationWrapper):
    """
    Resizes environment frames into a square image of size `shape`
    """
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def observation(self, observation):
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation
    
class SkipFrame(gym.Wrapper):
    """
    Skips n consecutive frames
    """
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

            