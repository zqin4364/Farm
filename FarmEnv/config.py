import argparse

def get_config():

    parser = argparse.ArgumentParser(
        description='FarmEnv', formatter_class=argparse.RawDescriptionHelpFormatter
    )
    #prepare parameters
    parser.add_argument('--algorithm-name', type=str, default='mappo', choices=['rmappo', 'mappo', 'hatrpo', 'maddpg'])

    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--cuda', action='store_false', default=True,
                        help='By default True, will use Gpu to train; or else will use CPU')
    parser.add_argument('--cuda_deterministic', action='store_false', default=True,
                        help='By default, make sure random seed effective. if set, bypass such function')
    parser.add_argument('--n_training_threads', type=int, default=1,
                        help='Number of torch threads for training')
    parser.add_argument('--n_rollout_threads', type=int, default=32,
                        help='Number of parallel envs for training rollouts')
    parser.add_argument('--n_eval_rollout_threads', type=int, default=1,
                        help='Number of parallel envs for evaluating rollouts')
    parser.add_argument('--n_render_threads', type=int, default=1,
                        help='Number of parallel envs for rendering rollouts')
    parser.add_argument('--num_env_steps', type=int, default=10e6,
                        help='Number of environment steps to train(default: 10e6)')

    #environment parameters
    parser.add_argument('--env_name', type=str, default='farm_v0', help='Specify the name of environment')
    parser.add_argument('--use_obs_instead_of_state', action='store_false', default=True, help='Whether to use global state or concatenated obs')

    #reply buffer parameters
    parser.add_argument('--episode_length', type=int, default=200, help='Max length for any episode')

    #estimator parameters
    parser.add


