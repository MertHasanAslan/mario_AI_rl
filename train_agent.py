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
from classes.saving import call_back

#create a preprocessed environment
def create_env_and_wrap():
    env = gym_super_mario_bros.make('SuperMarioBros-v0') #this will create a environment for super mario bros.
    env = JoypadSpace(env, SIMPLE_MOVEMENT) #now mario can do only his movements in the environment (right, left, jump...)

    ### since colored images (rgb) have 3 chanels it takes too many spaces
    ### but gray images only have 1 chanel so that it takes 1/3 spaces
    env = GrayScaleObservation(env, keep_dim = True) # make states gray

    env = DummyVecEnv([lambda: env]) #wrap env for making it vectorized (now it is in VecEnv format) (now you can use sb3)

    env = VecFrameStack(env, 4, channels_order='last') #last 4 observation

    return env


#train and save the model
def train_and_save():
    env = create_env_and_wrap()
    checkpoint_dir = './train/'
    log_dir = './logs/'

    callback = call_back(check_freq=10000, save_path=checkpoint_dir) #for saving AI model every 10000 steps

    # AI model which uses PPO algorithm for reinforcement learning
    #it will use CnnPolicy because this works with pictures very fast (frame = picture)
    #we will use pytorch cuda instead of cpu because it will make it so much faster.
    AI_model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_dir, learning_rate=0.000001, n_steps=512, device='cuda') 



    AI_model.learn(total_timesteps=1000000, callback=callback) #ai will start to learn

#you have to call directly this py, otherwise it won't work
if __name__ == "__main__": 
    train_and_save()

