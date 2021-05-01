'''
Single processed world model 

get CUDA working... issues with queue
get batch_size working... not worth it

human not moving while training rnn
'''

import argparse
from os import mkdir, unlink, listdir, getpid
from os.path import join, exists
from pickle import decode_long
from time import sleep, time
import sys
import random
import cma
import gym 
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm
import torch
import torch as pt
from torch import nn, optim, distributions
import matplotlib.pyplot as plt

# import torch.nn.functional as F
# from torch.autograd import Variable
from torch.multiprocessing import Process, Queue
import gym_gazeboros_ac



device = 'cpu' # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use_cuda = torch.cuda.is_available()

RANDOMSEED = 42  # random seed
torch.manual_seed(RANDOMSEED)
np.random.seed(RANDOMSEED)
torch.manual_seed(RANDOMSEED)

# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

###############################  hyper parameters  #########################
# ENV_NAME = 'gazeborosAC-v0'  # environment name
ENV_NAME = "LunarLander-v2"
batch_size = 1 # 128 # TODO fix https://github.com/ctallec/world-models/blob/master/trainmdrnn.py
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
LR = 0.001
# reward = 1
latent_space = 64 # hidden of the LSTM & of the controller 
# gaussian = 5

pop_size = 8 #4 # if this is larger than index, it will prevent a index error.
n_samples = 1 #4




parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="model_weights/world_model_"+ENV_NAME, type=str, help="Where everything is stored.")
parser.add_argument("--display", default=True, action="store_true", help="Use progress bars if specified.")
args = parser.parse_args()

time_limit = 500 #1000
print(f"args.logdir {args.logdir}")
print("First") # this gets called from every process, but not if its in main()... 
############################################################################

class RNN(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=latent_space, drop_prob=0.5):
        super(RNN, self).__init__()
        self.action_dim = act_dim
        self.observation_dim = obs_dim
        self.hidden = hid_dim
        # self.reward = reward
        self.learning_r = LR
        # self.gaussian_mix = gaussian
        # gmm_out = (2*obs_dim+1) * gaussian + 2

        self.rnn = nn.LSTMCell(obs_dim + act_dim, hid_dim)
        self.fc = nn.Linear(obs_dim + hid_dim, act_dim)
        self.mu = nn.Linear(hid_dim, obs_dim)
        self.logsigma = nn.Linear(hid_dim, obs_dim)

    def forward(self, obs, act, hid):
        x = torch.cat([act, obs], dim=-1)
        h, c = self.rnn(x, hid)
        mu = self.mu(h)
        # print("what is mu", mu)
        sigma = torch.exp(self.logsigma(h))
        # print("what is sigma", sigma)
        # print("this is the next state",c)
        # print("this is the next hidden",h)
        return mu, sigma, (h, c)

    def step(self, obs, h):
        # print(obs.size()) (1, 8)
        # print(h.size()) # (batch_size, 64)
        state = torch.cat([obs, h], dim=-1)
        # return torch.tanh(self.fc(state))
        return self.fc(state)


class Controller(nn.Module):
    """ Controller """

    def __init__(self, latents, actions, recurrents=latent_space):
        """[summary]

        Args:
            latents: [obs_dim]
            actions: [action_dim]
            recurrents: [Same as the hidden Dim of the LSTM]
        """
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, actions)

    def forward(self, obs, h):
        # print("we are in controller class?")
        # print(obs.size())
        # print(h.size())
        state = torch.cat([obs, h], dim=-1)
        return self.fc(state)




class RolloutGenerator(object):
    """Utility to generate rollouts.
    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.
    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """

    def __init__(self):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        # references: https://github.com/ctallec/world-models/blob/master/utils/misc.py
        ctrl_file = join(args.logdir, "ctrl", "best.tar")
        # obs_dim = 8 #TODO these need to be fixed for our environment
        # act_dim = 2   #2 #TODO these need to be fixed for out environment
        
        self.controller = Controller(obs_dim, act_dim).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={"cuda:0": str(device)})
            print(f"Loading Controller with reward {ctrl_state['reward']}")
            self.controller.load_state_dict(ctrl_state["state_dict"])
        else:
            print("\n\nController weights not found!\n\n")

    def get_action(self, obs):
        """

        """
        obs = env.reset() # CHANGED. we are now given it from rollout. 

        hid = (torch.zeros(1, latent_space), torch.zeros(1, latent_space))  # h  # c $ TODO 64 changed to latent_space, which is 64 

        obs = torch.from_numpy(obs).unsqueeze(0)
        act = self.controller.forward(obs, hid[0])
        act = torch.argmax(act)
        act = act.cpu().numpy()
        return act

    def rollout(self, params, render=False):
        """Execute a rollout and returns minus cumulative reward.
        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.
        :args params: parameters as a single 1D np array
        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)
        # print(f'we are in rollout')
        obs = env.reset()
        cumulative = 0
        i = 0
        
        while True:
            # TODO in the past get_action() has env.reset inside it... but we step below..this might has been a mistake
            action = self.get_action(obs) 
            # print(action)
            obs, reward, done, _ = env.step(action)

            # if render or i == 0:  # This first render is required! # TODO is it? 
            #     env.render()
            #     pass

            cumulative += reward

            if done or i > time_limit:
                env.close()  # TODO added causing problums in windows thread exiting
                return -cumulative
            i += 1
        


def unflatten_parameters(params, example, device):
    """Unflatten parameters.
    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters
    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    # print(f'params.size is {params.size()}\n\n')
    for e_p in example:
        unflattened += [params[idx : idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened


def flatten_parameters(params):
    """Flattening parameters.
    :args params: generator of parameters (as returned by module.parameters())
    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()


def load_parameters(params, controller):
    """Load flattened parameters into controller.
    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

def evaluate(solutions, results, rollouts=100):
    """Give current controller evaluation.
    Evaluation is minus the cumulated reward averaged over rollout runs.
    :args solutions: CMA set of solutions
    :args results: corresponding results
    :args rollouts: number of rollouts
    :returns: minus averaged cumulated reward
    """
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    for s_id in range(rollouts):
        p_queue.put((s_id, best_guess))

    print("Start Evaluating...")
    # for _ in tqdm(range(rollouts)):
    #         sleep(0.1)
    while not r_queue.empty():
        restimates.append(r_queue.get()[1])
    print("Done Evaluating...")
    return best_guess, np.mean(restimates), np.std(restimates)

# Thread routines 
def slave_routine():
    print(f'in slave_routine')
    """Thread routine.
    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.
    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.
    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).
    """
    # with torch.no_grad():
    while not p_queue.empty():
        # sleep(0.5) # waiting wont do us any good. single thread
        # print("p_queue is not empty. We are putting stuff in r_queue")
        s_id, params = p_queue.get()
        r_queue.put((s_id, rollout_generator.rollout(params)))  
    print("p_queue is empty")




if __name__ == "__main__":
    #TODO: anthony said to append obstacles onto state -> to get laser scan!
    #Anthony made changes to gym environment

    env = gym.make(ENV_NAME)
    #env.seed(RANDOMSEED)
    # env.set_agent(0)
    
    obs_dim = env.observation_space.shape[0]  # this is for our environment -> 67 dimensions: 47 = system state + 20 = laser scan
    # act_dim = env.action_space.shape[0]
    act_dim = env.action_space.n
    print(f"obs_dim is {obs_dim} and  act_dim is {act_dim}")


    ctrl_dir = join(args.logdir, "ctrl")
    if not exists(ctrl_dir):
        mkdir(ctrl_dir)
      
    rnn_dir = join(args.logdir, "rnn")
    if not exists(rnn_dir):
        mkdir(rnn_dir)

    rnn_filename = rnn_dir + "/my_rnn.pt"
    ctrl_filename = ctrl_dir + "/best.tar"


    model = RNN(obs_dim, act_dim)
    try:
        model.load_state_dict(pt.load(rnn_filename))
        print(f"LSTM weights loaded")
    except Exception as e:
        print(f"Error in loading weights, file might not be found or model may have changed\n{e}\n\n")
        
    model.zero_grad()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.RMSprop(model.parameters())
    criterion = torch.nn.MSELoss()
    # softmax = nn.Softmax(dim=1)
    
    losses, rewards = [], []
    num_episodes = 100      # 
    episode_length = 99999  # 
    print("#################### LETS DO THE RNN ###############################")

    epoch_count = []
    #TODO: IMPORTANT: ANTHONY SAID TO MAKE SURE EVERY INNER LOOP  15 SECONDS -> 
    #Suggestion: ANTHONY -> said to do a data collection before training
    
    for i_episode in range(num_episodes): 
        # Initialize the environment and state
        state = env.reset()
        #state = torch.Tensor([state]) #this might have to changed depending on how the state is structured
        state = torch.from_numpy(np.array([state], dtype=np.float32))
        hid = (torch.zeros(batch_size, model.hidden, dtype=pt.float), # .to(device)
               torch.zeros(batch_size, model.hidden, dtype=pt.float))
        epoch_count.append(i_episode)
        loss = 0.0
        now = time()
        for i in range(episode_length):
            # state = state #.to(device=device)
            pred = model.step(state, hid[0])
            #TODO: action goal is a x, y vector that is translative to the robot
            #Important to acknowledge: Obstacle Avoidance -> if for some reason this doesnt work -> we might want our thing to output a linear and angular velocity. -> we will have to transition that to the robot-robot using the TEB
            
            # Continous
            # action = pred.detach().clamp(min=-1, max=1).numpy().flatten()  # ensure bounds
            # Discrete
            action = torch.argmax(pred).numpy()


            print(f'action = {action}')
            mu, sigma, hid = model.forward(state, pred, hid)

            #sleep(0.1) #TODO ANTHONY SAID TO INSERT SLEEP 
            state, reward, done,_  = env.step(action) #not sure what the data structure is  #TODO need to to put variable instead of underscore!!! 


            # print(f'state before {state}')
            # print(f'state mid    {np.array([state)}')
            # TODO #this might have to changed depending on how the state is structured -> make more efficient --> the next state is this and we use this next state to calcualte the log_prob in the loss function
            state = torch.from_numpy(np.array([state], dtype=np.float32)) 


            #print("this si the next state", state)
            dist = distributions.Normal(loc=mu, scale=sigma)
            nll = -dist.log_prob(state)  # negative log-likelihood of the next state!
            nll = torch.mean(nll, dim=-1)   # mean over dimensions # TODO use mean or sum? 
            nll = torch.mean(nll, dim=0)    # mean over batch
            loss += nll
            # print(done)
            if done:
                break

        # print(f"time taken = {time() - now}")
        # print(f'val  is {val}')
        print(f'reward  is {reward}')
        loss = loss/episode_length 
        val = loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # state = torch.tensor(state)
        # print(f'state after {state}')
        # print("#################### UPDATED STATE #########################")
        losses.append(val)
        rewards.append(reward)
    torch.save(model.state_dict(), rnn_filename)

    # env.close()
    # print('close env and sleep')
    # sleep(3)
    # env = gym.make(ENV_NAME)
    # #env.seed(RANDOMSEED)
    # env.set_agent(0)
    # print('now we have a new env')

    #plt.plot(epoch_count, losses)
    #plt.show()
    
    cur_best = None

    p_queue = Queue() 
    # filled inside evaulate and in main controller loop 
    # emptied in slave

    r_queue = Queue() 
    # filled in slave. 
    # emptied evaulate and in main controller loop 

    print("#################### QUEUES ARE INITIALIZED #####################")

    print(f'obs_dim, act_dim = {obs_dim} |  {act_dim}')
    controller = Controller(obs_dim, act_dim) # dummy instance
    # try:
    #     saved_data = torch.load(ctrl_filename) 
    #     cur_best = -saved_data['reward']
    #     controller.load_state_dict(pt.load(saved_data["state_dict"]))
    #     print(f"controller weights loaded")
    # except Exception as e:
    #     print(f"Error in loading controller weights, file might not be found or model may have changed\n{e}\n\n")

    print("#################### CONTROLLER SETUP DONE #########################")
    parameters = controller.parameters()
    es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1, {"popsize": pop_size})
    rollout_generator = RolloutGenerator() # global variable for use in slave routine

    target_return = 250
    epoch = 0
    log_step = 3
    cur_best = 0

    print("#################### ABOUT TO RUN CONTROLLER TRAINING ################")
    while True: # notes.stop(): #  we could* make this True
        if -cur_best > target_return:
            print("Already better than target, terminating...")
            break
        result_list = [0] * pop_size  # result list. like np.zeros(pop_size).tolist()
        solutions = es.ask()

        # push parameters to queue
        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.put((s_id, s))

        # This slave call is stealing the data the other slave calls needs..
        if epoch % log_step != 0:
            slave_routine() # fill r_queque with p_queue WITH IS FROM ABOVE 

        # print("we just put something in p_queue")
        
        while not r_queue.empty():
            # print("We are in this for loop?")
            result_list_idx, r = r_queue.get()
            try:
                result_list[result_list_idx] += r / n_samples
                # print(f'r_queue is not empty', result_list)
            except Exception as e:
                print(f'result_list_idx is {result_list_idx}')
                print(f'Caught error. {e}')



        es.tell(solutions, result_list)
        es.disp()

        # evaluation and saving
        if epoch % log_step == 0:
            slave_routine() # fill r_queque with p_queue, WHICH IS FROM evaluate()
            best_params, best, std_best = evaluate(solutions, result_list)
            print(f"Current evaluation: {best}+/-{std_best}") # :.2f
            if not cur_best or cur_best > best:
                cur_best = best
                print(f"Saving... Current best is {cur_best}")
                load_parameters(best_params, controller)
                torch.save(
                    {"epoch": epoch,
                     "reward": -cur_best, # TODO why do we have a negate?  https://github.com/ctallec/world-models/blob/master/traincontroller.py#L203
                     "state_dict": controller.state_dict(), },
                        join(ctrl_dir, "best.tar"))
            if -best > target_return:
                print(f"Terminating controller training with value {best}...")
                break

        epoch += 1
        print(f'starting epoch: {epoch}. es.stop() is {es.stop()}')
        
    print('program exiting...')
    es.result_pretty()
    env.close()
