## Project Struct  
  FarmEnv/  
    ├── algorithm/ # 强化学习算法模块  
    │ ├── utils/  
    │ │ └── init.py  
    │ └── ppo.py # PPO 算法实现  
    │  
    ├── envs/ # 多智能体环境模块  
    │ ├── utils/  
    │ │ └── init.py  
    │ ├── config.py # 环境配置  
    │ ├── core.py # 环境核心逻辑（World、Agent 等）  
    │ └── wrapper.py # 环境包装器（如 PettingZoo 封装）  
    │  
    ├── result/ # 实验结果保存目录  
    │  
    ├── train/ # 训练脚本目录  
    │  
    ├── utils/ # 通用工具模块  
    │ ├── init.py  
    │ └── config.py # 全局配置  
    │  
    ├── requirements.txt # 依赖包列表  
    └── test.py # 测试脚本  
