from gymnasium.envs.registration import  register
from FarmEnv.envs.wrapper import FarmEnvWrapper

register(
    id="Farm-v0",
    entry_point="FarmEnv.envs.wrapper: FarmEnvWrapper",
    kwargs={
        "num_farmers":3,
        "total_res":100,
        "max_step":50
    }
)