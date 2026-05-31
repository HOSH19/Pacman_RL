from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

import argparse
from datetime import datetime


# This script should have some options
# 1. Turn off the stochasticity as determined by the ALEv5
#   Even if deterministic is set to true in evaluate policy, the environment will ignore this 25% of the time
#   To compensate for this, we can set the repeat action probability to 0
#       DONE
# 2. Print out the evaluation metrics or save to file
#       DONE
# 4. Print the keyword args for the environment? I think this might be helpful...
#       DONE (ish), printing the environment specifications.
# 5. Add option flag to accept file path for model
#       DONE
# 6. Add option flag to accept number of episodes
#       DONE
# 7. Save evaluations in a log file
#       DONE
# 8. Add option flag for mean rewards/length or discrete rewards/lengths
#       IN PROGRESS

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--repeat_action_probability", help="repeat action probability, default 0.25", type=float, default=0.25)
parser.add_argument("-f", "--frameskip", help="frameskip, default 4", type=int, default=4)
# parser.add_argument("-o", "--observe", help="observe agent", action="store_const", const=True)
parser.add_argument("-p", "--print", help="print environment information", action="store_const", const=True)
parser.add_argument("-e", "--num_episodes", help="specify the number of episodes to evaluate, default 1", type=int, default=1)
parser.add_argument("-a", "--agent_filepath", help="file path to agent to watch, minus the .zip extension", type=str, required=True)
# parser.add_argument("-s", "--savefile", help="Specify a filepath to save the evaluation metrics.", type=str, default="evals")
args = parser.parse_args()

model_name = args.agent_filepath
model = DQN.load(model_name)
# There should really be a condition here to catch input defining directories with forward slashes
dirs = model_name.split("/")
# remove the last item, as it is the zip file
dirs.pop()
model_dir = "/".join(dirs)

# Retrieve the environment
eval_env = Monitor(gym.make("ALE/Pacman-v5", 
                            render_mode="rgb_array", 
                            repeat_action_probability=args.repeat_action_probability,
                            frameskip=args.frameskip))

if args.print == True:
    env_info = str(eval_env.spec).split(", ")
    for item in env_info:
        print(item)
# Evaluate the policy
# Toggle the mean or discrete evaluations here
mean_rwd, std_rwd = evaluate_policy(model.policy, eval_env, n_eval_episodes=args.num_episodes)

# savefile = args.savefile
savefile = model_dir + "/evals"
date = datetime.now().strftime("%d %b %Y")
time = datetime.now().strftime("%I:%M:%S %p")

with open(f"{savefile}.txt", "a") as file:
    file.write("-----\n")
    file.write(f"Evaluation of {model_name} on {date} at {time}\n")
    file.write(f"Episodes evaluated: {args.num_episodes}\n")
    file.write(f"mean_rwd: {mean_rwd}\n")    
    file.write(f"std_rwd: {std_rwd}\n\n")
    
