![Isaac Lab](docs/source/_static/pipeline_main.jpg)


## Getting Started
The installation process fully complies with the official Isaac Lab documentation, with the exception that you need to clone the current pipeline, and not from the official Isaac Lab repository with Isaac sim 4.5

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)

The launch is carried out in accordance with the official documentation. The name of the environment - Isaac-Aloha-Direct-v0
The SAC algorithm of the skrl library is used here.

For Imitation Learning to work correctly, you need to transfer and copy the contents of the skrl folder to the appropriate folder using the approximate path "/home/${HOME}/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/skrl"

Train:
```
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Aloha-Direct-v0 --num_envs 32 --enable_cameras --headless
```

Play:
```
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Aloha-Direct-v0 --algorithm SAC --num_envs 1 --checkpoint /home/xiso/IsaacLab/logs/skrl/aloha/2025-08-11_09-55-36_ppo_torch_SAC/checkpoints/agent_14000.pt --enable_cameras
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
Asset directories should look like this:
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
