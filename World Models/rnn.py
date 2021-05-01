'''
What  I haven't bothered TODO
 - get CUDA working... issues with queue
 - get batch_size working... not worth it
'''

import argparse
from os import mkdir, unlink
from os.path import join, exists
from pickle import decode_long
from time import sleep, time
import sys
import cma
import gym 
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm
import torch
from torch import nn, optim, distributions

# import torch.nn.functional as F
# from torch.autograd import Variable
from torch.multiprocessing import Process, Queue
import gym



device = 'cpu' # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use_cuda = torch.cuda.is_available()

RANDOMSEED = 42  # random seed
torch.manual_seed(RANDOMSEED)
np.random.seed(RANDOMSEED)



###############################  hyper parameters  #########################
ENV_NAME = "LunarLander-v2"
SLEEP_TIME = 0.3

batch_size = 1 # 128 # TODO fix https://github.com/ctallec/world-models/blob/master/trainmdrnn.py
latent_space = 128 # hidden of the LSTM & of the controller 

num_episodes = 1

pop_size = 4
n_samples = 4

# num_workers if you change from 1, then you'll have to fix env.set_agent which is hardcoded :) 
num_workers = 1 # 32 # not sure if any benefit from more than 3 workers 
num_workers = min(num_workers, n_samples * pop_size) 


parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="model_weights\\world_model_"+ENV_NAME, type=str, help="Where everything is stored.")
args = parser.parse_args()

time_limit = 1000
print(f"args.logdir {args.logdir}")
# print("First") # this gets called from every process, but not if its in main()... 
############################################################################


class RNN(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=latent_space, drop_prob=0.5):
        super(RNN, self).__init__()
        self.action_dim = act_dim
        self.observation_dim = obs_dim
        self.hidden = hid_dim

        # LSTM can have num_layers=2
        self.rnn_1 = nn.LSTMCell(input_size = obs_dim + act_dim, hidden_size=hid_dim)
        self.rnn_2 = nn.LSTMCell(input_size=hid_dim, hidden_size=hid_dim)
        # self.linear = nn.Linear(hid_dim, hid_dim)
        self.mu = nn.Linear(hid_dim, obs_dim)
        self.logsigma = nn.Linear(hid_dim, obs_dim)
        
        self.fc = nn.Linear(obs_dim + hid_dim, act_dim)


    def forward(self, obs, act, hid):
        x = torch.cat([act, obs], dim=-1)
        h, c = self.rnn_1(x, hid)
        h, c = self.rnn_2(h)
        # h = self.linear(h) 
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
        print(f'Controller maps {latents + recurrents} to {actions}, which we argmax over')
        self.fc = nn.Linear(latents + recurrents, actions)

    def forward(self, obs, h):
        # print(f"obs.size {obs.size()} | h.size {h.size()}")
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

    def __init__(self, mdir, rnn, device, time_limit):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        # references: https://github.com/ctallec/world-models/blob/master/utils/misc.py
        ctrl_file = join(mdir, "ctrl", "best.tar")
        # obs_dim = 50  #8 #TODO these need to be fixed for our environment
        # act_dim = 2   #2 #TODO these need to be fixed for out environment
        # obs_dim = self.env.observation_space.shape[0]
        # act_dim = env.action_space.shape[0]
        self.model = rnn
        # self.model.load_state_dict({k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
        self.controller = Controller(obs_dim, act_dim).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={"cuda:0": str(device)})
            print(f"Loading Controller with reward {-ctrl_state['reward']}")
            self.controller.load_state_dict(ctrl_state["state_dict"])
        else:
            print("\n\nController weights not found!\n\n")

        self.env = gym.make(ENV_NAME)
        # self.env.set_agent(1)
        self.device = device

        self.time_limit = time_limit

    def get_action_and_transition(self, seq_len=1600):
        """Get action and transition.
        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.
        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor
        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        obs = self.env.reset()
        hid = (torch.zeros(1, latent_space), torch.zeros(1, latent_space))  # h  # c $ TODO 64 changed to latent_space, which is 64 
        obs = torch.from_numpy(obs).unsqueeze(0)
        # print(f'hid[0] {hid[0].size()}  | obs is {obs.size()}')
        act = self.controller.forward(obs, hid[0])
        _, _, hid = self.model.forward(obs, act, hid)
        # m = nn.Softmax(dim=1)
        # act = m(act)
        act = torch.argmax(act)
        act = act.cpu().numpy()
        # _, latent_mu, _ = self.vae(obs)
        # action = self.controller(latent_mu, hidden[0])
        # _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        return act, obs

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
        obs = self.env.reset() # was self.
        cumulative = 0
        i = 0
        

        while True:
            action, hidden = self.get_action_and_transition()
            obs, reward, done, _ = self.env.step(action)

            if render:
                self.env.render()

            cumulative += reward

            if done or i > self.time_limit:
                self.env.close()  # TODO added causing problums in windows thread exiting
                return -cumulative
            i += 1
        

################################################################################
#                           Thread routines                                    #
################################################################################
def slave_routine(p_queue, r_queue, e_queue, p_index, model):
    """Thread routine.
    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue and e_queue the end queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.
    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.
    As soon as e_queue is non empty, the thread terminate.
    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).
    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """
    # init routine
    # gpu = p_index % torch.cuda.device_count()
    # device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # print("we are in slave_routine")
    # redirect streams
    # sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
    # sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')
    # print(p_queue)
    with torch.no_grad():
        r_gen = RolloutGenerator(args.logdir, model, device, time_limit)
        while e_queue.empty():
            if p_queue.empty():
                # print("we are in if statement")
                sleep(SLEEP_TIME)
            else:
                # print("we are in else statement")
                s_id, params = p_queue.get()
                # print("we are putting stuff in r_queue")
                r_queue.put((s_id, r_gen.rollout(params)))


################################################################################
#                           Controller                                         #
################################################################################
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


ctrl_dir = join(args.logdir, "ctrl")
if not exists(ctrl_dir):
    mkdir(ctrl_dir)
        
rnn_dir = join(args.logdir, "rnn")
if not exists(rnn_dir):
    mkdir(rnn_dir)


## Global
env = gym.make(ENV_NAME)
#env.seed(RANDOMSEED)

obs_dim = env.observation_space.shape[0] 
# act_dim = env.action_space.shape[0]
act_dim = env.action_space.n

if __name__ == "__main__":


    print("obs_dim + act_dim", obs_dim + act_dim)
    model = RNN(obs_dim, act_dim)
    rnn_filename = rnn_dir + "/my_rnn.pt"
    ctrl_filename = ctrl_dir + "/best.tar"
   
    try:
        model.load_state_dict(torch.load(rnn_filename))
        print(f"LSTM weights loaded")
    except Exception as e:
        print(f"Error in loading weights, file might not be found or model may have changed\n{e}\n\n")
        
    model.zero_grad()
    model.to(device)

    # params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.RMSprop(model.parameters())
    criterion = torch.nn.MSELoss()
    # softmax = nn.Softmax(dim=1)
    
    losses = []
    rewards = [] 
    epoch_count = []
    
    print("#################### LETS DO THE RNN ###############################")


    #TODO: IMPORTANT: ANTHONY SAID TO MAKE SURE EVERY INNER LOOP  15 SECONDS -> 
    #Suggestion: ANTHONY -> said to do a data collection before training
    
    for i_episode in range(num_episodes): 
        # Initialize the environment and state
        state = env.reset()
        #state = torch.Tensor([state]) #this might have to changed depending on how the state is structured
        state = torch.from_numpy(np.array([state], dtype=np.float32))
        hid = (torch.zeros(batch_size, model.hidden, dtype=torch.float), # .to(device)
               torch.zeros(batch_size, model.hidden, dtype=torch.float))
        epoch_count.append(i_episode)
        loss = 0.0 
        now = time()
        episode_length = 0 
        while True:
            # state = state #.to(device=device)
            pred = model.step(state, hid[0]).detach()
                        
            # Continous
            # action = pred.clamp(min=-1, max=1).numpy().flatten()  # ensure bounds
            
            # Discrete
            action = torch.argmax(pred).numpy()

            # print(f'action = {action}')
            mu, sigma, hid = model.forward(state, pred, hid)

            #sleep(SLEEP_TIME) #TODO ANTHONY SAID TO INSERT SLEEP 
            state, reward, done,_  = env.step(action)


            # print(f'state before {state}')
            # print(f'state mid    {np.array([state)}')
            # this might have to changed depending on how the state is structured -> make more efficient --> the next state is this and we use this next state to calcualte the log_prob in the loss function
            state = torch.from_numpy(np.array([state], dtype=np.float32)) 


            dist = distributions.Normal(loc=mu, scale=sigma)
            nll = -dist.log_prob(state)     # negative log-likelihood of the next state!
            nll = torch.mean(nll, dim=-1)   # mean over dimensions # TODO use mean or sum? 
            nll = torch.mean(nll, dim=0)    # mean over batch
            loss += nll
            # print(done)
            episode_length+=1
            if done:
                break

        # print(f"")
        loss = loss/episode_length 
        val = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch is {i_episode} | Rewards is {reward} | Loss: {val:.6f} | time taken = {(time() - now):.6f}')

        losses.append(val)
        rewards.append(reward)

        if i_episode % 200: 
            torch.save(model.state_dict(), rnn_filename)

    # save rewards and weights to disk
    with open("lstm_reward.txt", "a") as f:
        for reward in rewards:
            f.write(f" {reward},")
    torch.save(model.state_dict(), rnn_filename)

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    window_size = 50 
    # plt.plot(moving_average(rewards, window_size))
    # plt.xlabel('Episode')
    # plt.ylabel(f'Rewards (MA-{window_size})')
    # plt.title(f'Rewards of World Model\'s LSTM')
    # plt.savefig('Rewards with MA.png')
    # plt.show()
    # plt.clf()
    # plt.cla()
    # plt.close()

    # plt.plot(moving_average(losses, window_size))
    # plt.xlabel('Episode')
    # plt.ylabel(f'Loss (MA-{window_size})')
    # plt.title(f'Loss of World Model\'s LSTM')
    # plt.savefig('Loss with MA.png')
    # plt.show()
    # plt.clf()
    # plt.cla()
    # plt.close()
    
    

    p_queue = Queue()
    r_queue = Queue()
    e_queue = Queue()

    print("#################### QUEUES ARE INITIALIZED #####################")

    for p_index in range(num_workers):
        Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index, model)).start()

    print("#################### PROCESSING COMPLETE #######################")

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

        print("Evaluating...")
        for _ in tqdm(range(rollouts)):
            while r_queue.empty():
                sleep(SLEEP_TIME)
            restimates.append(r_queue.get()[1])

        return best_guess, np.mean(restimates), np.std(restimates)



    print("#################### LET US SETUP #################################")
    print(f'obs_dim, act_dim = {obs_dim} |  {act_dim}')
    controller = Controller(obs_dim, act_dim) # dummy instance


    print("#################### CONTROLLER SETUP ##########################")
    parameters = controller.parameters()
    es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), SLEEP_TIME, {"popsize": pop_size})

    target_return = 350
    epoch = 0
    log_step = 3
    cur_best = None

    print("#################### ABOUT TO RUN CONTROLLER TRAINING ####################")
    while not es.stop():
        if cur_best is not None and -cur_best > target_return:
            print("Already better than target, breaking...")
            break

        r_list = [0] * pop_size  # result list. like np.zeros(pop_size).tolist()
        solutions = es.ask()

        # push parameters to queue
        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.put((s_id, s))

        # print("we just put something in p_queue")
        


        for _ in tqdm(range(pop_size * n_samples)):
            # print("We are in this for loop?")
            while r_queue.empty():
                sleep(SLEEP_TIME)
            r_s_id, r = r_queue.get()
            r_list[r_s_id] += r / n_samples


        es.tell(solutions, r_list)
        es.disp()

        # evaluation and saving
        if epoch % log_step == log_step - 1:
            best_params, best, std_best = evaluate(solutions, r_list)
            print(f"Current evaluation: {-best}+/-{std_best}") # :.2f
            if not cur_best or cur_best > best: # We want to minimize, this is correct 
                cur_best = best
                print(f"[Saving] New Best evaluation: {-cur_best}+/-{std_best}")
                load_parameters(best_params, controller)
                torch.save(
                    {"epoch": epoch,
                     "reward": -cur_best, # TODO why do we have a negate?  https://github.com/ctallec/world-models/blob/master/traincontroller.py#L203
                     "state_dict": controller.state_dict(), },
                        join(ctrl_dir, "best.tar"))
            if -best > target_return:
                print(f"Terminating controller training with value {-best}...")
                break

        epoch += 1
        
    print('program exiting...')
    es.result_pretty()
    e_queue.put("EOP")
    env.close()
    
