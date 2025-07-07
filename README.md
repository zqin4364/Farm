Struct Tree

Farm/
├── FarmEnv/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── core.py              # definition of the main environment(including entities, step logic, class World, etc).
│   │   ├── wrapper.py           # PettingZoo interface encapsulation.
│   │   ├── config.py            
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── true_model.py    # True_Demand function
│   │       ├── bayesian_estimator.py  # bayesian_estimator
│   │       └── credit.py        # 信用机制（若有）
│
├── train/
│   ├── __init__.py
│   ├── train_v0_ippo.py         # 使用 IPPO 训练的主脚本
│   └── utils.py                 # 训练辅助函数（如可视化、日志记录）
│
├── scripts/                     # 可选：训练/评估的 bash 或 bat 脚本
│   └── run_train.sh
│
├── checkpoints/                # 模型保存路径
│   └── ippo_model.pt
│
├── logs/                       # 训练日志和 Tensorboard 文件
│   └── ...
│
├── README.md
└── requirements.txt
