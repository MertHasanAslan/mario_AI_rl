import gym_super_mario_bros #import the mario game
from nes_py.wrappers import JoypadSpace #import joypad wrapper
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT #import actions of mario
import cv2 #import opencv so that we can resize the screen
from gym.wrappers import FrameStack, GrayScaleObservation #for making states gray (rgb = 3gray)
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv #for converting env to vector and giving last couple frames to agent
from matplotlib import pyplot as plt #for visuliation graphs etc.
import os
from stable_baselines3 import PPO #a reinforcement learning algo that we are using for this game
from stable_baselines3.common.callbacks import BaseCallback #for saving AI models
#do not have to import torch because sb3 already using pytorch.

class call_back(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(call_back, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok = True)

    def _on_step(self):
        if self.n_calls % self.check_freq ==0:
            model_path = os.path.join(self.save_path, 'best_model{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
        
