![Isaac Lab](docs/source/_static/pipeline_main.jpg)


## Getting Started
The installation process fully complies with the official Isaac Lab documentation, with the exception that you need to clone the current pipeline, and not from the official Isaac Lab repository.

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)

The launch is carried out in accordance with the official documentation. The name of the environment - Isaac-Aloha-Direct-v0
The SAC algorithm of the skrl library is used here.

link for assets:
```
https://drive.google.com/drive/folders/1JpMy9bIAhsimuZCMyoscgRr9rP1W7ewi?usp=drive_link
```

Train:
```
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Aloha-Direct-v0 --num_envs 32 --enable_cameras --headless
```
Train without graph:
```
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Aloha-Direct-vIMG --num_envs 32 --enable_cameras --headless
```
Train with graph:
```
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Aloha-Direct-vG --num_envs 32 --enable_cameras --headless
```
Play:
```
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Aloha-Direct-v0 --algorithm SAC --num_envs 1 --checkpoint /home/xiso/IsaacLab/logs/skrl/aloha/2025-08-11_09-55-36_ppo_torch_SAC/checkpoints/agent_14000.pt --enable_cameras
```
If you want check inference with metrics set in aloha_env.py:
```
self.eval = True
```
If you want record video you should add:
```
--video --video_length 512
```

Imitation learning:
For IL firsly you should get paths:
```
./isaaclab.sh -p path_generator.py
```
To generate paths via dijkstra algo (Check, that there no all_paths.json in data):
```
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/aloha/path_generator.py 
```

Asset directories should look like this by the path IsaacLab/source/isaaclab_assets/data/:
```
└── aloha_assets
    ├── aloha
    │   ├── aloha.usd
    │   └── realsense.usd
    ├── objects
    │   └── bowl.usd
    └── scenes
        ├── obstacles
        └── scenes_sber_kitchen_for_BBQ
            ├── kitchen_new_simple.usd
            └── table
```

Also it is nessesery replace the folder "skrl" on the path miniconda3/envs/env_isaaclab/lib/python3.10/site-packages to "skrl" in pipeline