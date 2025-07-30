## 1.Project Struct  
  FarmEnv/  
    ├── algorithm/ 
    │ ├── utils/  
    │ │ └── init.py  
    │ └── ddpg.py
    │  
    ├── envs/ # envs
    │ ├── utils/  
    │ │ └── init.py  
    │ ├── config.py # param of envs 
    │ ├── core.py 
    │ └── wrapper.py # wrapper to petting zoo  
    │  
    ├── result/ 
    │  
    ├── train/  
    │  
    ├── utils/  
    │ ├── init.py  
    │ └── config.py   
    │  
    ├── requirements.txt  
 
## 2.Installation  
conda create -n farm python==3.10.16
conda activate farm
cd FarmEnv

install farm package
pip install -e

