# Modified from Isaac Gym Environments for Legged Robots #

### Installation ###
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 3 (does not directly work on Preview 4)
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`
4. Install modified rsl_rl (PPO implementation) and legged_gym
   - Clone and `cd` to this repository
   -  `cd rsl_rl && pip install -e .` 
   - `cd legged_gym_dance && pip install -e .`
5. Install other dependencies
   - `pip install tensorboard wandb`

### Usage ###
0. ```cd legged_gym_dance```
1. Train:  
  ```python legged_gym/scripts/train.py --task=cyber2_stand_dance --headless```
    - The trained policy is saved in `legged_gym/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    -  The following command line arguments override the values set in the config files:
     - --task TASK: Task name.
     - --resume:   Resume training from a checkpoint
     - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
     - --run_name RUN_NAME:  Name of the run.
     - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
     - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
     - --num_envs NUM_ENVS:  Number of environments to create.
     - --seed SEED:  Random seed.
     - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
2. Play a trained policy:  
```python legged_gym/scripts/play.py --task=cyber2_stand_dance --load_run 2023-09-01-12-57-42_initsit_sliph0.03-0.4_qdiff-1_shift-50_vec0.2_resetpos5deg_initcontact30_com0.01_wvel --checkpoint 18000 --headless```
