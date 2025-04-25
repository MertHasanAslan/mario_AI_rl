import cv2
from train_agent import create_env_and_wrap
from stable_baselines3 import PPO

def show_resized_random(env):
    done = True #this boolean will check if game is end or not

    for step in range(10000):
        if done:
            state = env.reset()

        state, reward, done, info = env.step([env.action_space.sample()]) #make a random move and return state, reward, done, info

        #not doing this part will make process very very much slower.
        #If you want to avoid this, use simply "env.renger()" rather then this converting
        frame = env.render(mode = 'rgb_array') #frame = env.render but in [(red, blue, green)...] array
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #since we took rgb array. we have to convert it to bgr array.
        frame_resized = cv2.resize(frame_bgr, (800, 600)) #now it will display screen in 800x800pixel
        cv2.imshow("Super Mario Bros AI", frame_resized) #display

        if cv2.waitKey(1) & 0xFF == ord('q'):  #waitkey means fps and if you press q screen will close
            break
    #close environment and screen
    env.close() 
    cv2.destroyAllWindows()

def show_random(env):
    done = True #this boolean will check if game is end or not

    for step in range(10000):
        if done:
            state = env.reset()

        state, reward, done, info = env.step([env.action_space.sample()]) #make a random move and return state, reward, done, info

        env.render()
    
    env.close() 


def show_agent(path_of_model):
    env = create_env_and_wrap()  #create environment
    model = PPO.load(path_of_model) #load AI_model

    state = env.reset() #start the game

    if isinstance(state, tuple):
        state = state[0]

    _env = env.envs[0].env  

    while True:
        action, _state = model.predict(state)
        state, reward, done, info = env.step(action)
        if isinstance(state, tuple):
            state = state[0]

        frame = _env.render(mode="rgb_array")  
        if frame is not None: #prevent from crashing
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_bgr, (800, 600))
            cv2.imshow("Super Mario Bros", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'): #q = break game
            break

    env.close()
    cv2.destroyAllWindows()

