import gym_super_mario_bros #import the mario game
from nes_py.wrappers import JoypadSpace #import joypad wrapper
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT #import actions of mario
import cv2 #import opencv so that we can resize the screen

env = gym_super_mario_bros.make('SuperMarioBros-v0') #this will create a environment for super mario bros.
env = JoypadSpace(env, SIMPLE_MOVEMENT) #now mario can do only his movements in the environment (right, left, jump...)


done = True #this boolean will check if game is end or not

for step in range(10000):
    if done:
        state = env.reset()

    state, reward, done, info = env.step(env.action_space.sample()) #make a random move and return state, reward, done, info

    frame = env.render(mode = 'rgb_array') #frame = env.render but in [(red, blue, green)...] array
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #since we took rgb array. we have to convert it to bgr array.
    frame_resized = cv2.resize(frame_bgr, (800, 600)) #now it will display screen in 800x800pixel
    cv2.imshow("Super Mario Bros", frame_resized) #display

    if cv2.waitKey(1) & 0xFF == ord('q'):  #waitkey means fps and if you press q screen will close
        break


#close environment and screen
env.close() 
cv2.destroyAllWindows()
