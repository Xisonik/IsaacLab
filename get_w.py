import torch

# Загружаем чекпоинт
path1 = "/home/xiso/IsaacLab/logs/skrl/aloha/2025-08-10_11-59-04_ppo_torch_SAC/checkpoints/agent_500.pt"
path2 = "/home/xiso/IsaacLab/logs/skrl/aloha/2025-08-10_11-59-04_ppo_torch_SAC/checkpoints/agent_1000.pt"
path3 = "/home/xiso/IsaacLab/logs/skrl/aloha/2025-08-10_11-59-04_ppo_torch_SAC/checkpoints/agent_2000.pt"
ckpts = []
ckpt = torch.load(path1, map_location="cpu")
ckpts.append(ckpt)
ckpt = torch.load(path2, map_location="cpu")
ckpts.append(ckpt)
ckpt = torch.load(path3, map_location="cpu")
ckpts.append(ckpt)
for ckpt in ckpts:
    for block_name, block_params in ckpt.items():
        if isinstance(block_params, dict) and block_name == "policy":
            print(f"=== {block_name} ===")
            for param_name, tensor in block_params.items():
                print(f"{param_name}: {tensor.shape} min={tensor.min().item():.4f} max={tensor.max().item():.4f}")
        else:
            pass
            # print(f"{block_name}: {type(block_params)}")

# def compare_policy(path1, path2):
ckpt1 = torch.load(path1, map_location="cpu")["policy"]
ckpt2 = torch.load(path2, map_location="cpu")["policy"]

for name in ckpt1:
    t1, t2 = ckpt1[name], ckpt2[name]
    if torch.is_tensor(t1) and torch.is_tensor(t2):
        diff = (t1 - t2).abs().max().item()
        print(f"{name}: max abs diff = {diff:.6e}")

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python compare_policy.py model1.pt model2.pt")
#         sys.exit(1)

#     compare_policy(sys.argv[1], sys.argv[2])


# ckpt = torch.load("/home/xiso/IsaacLab/logs/skrl/aloha/2025-08-10_11-59-04_ppo_torch_SAC/checkpoints/agent_1000.pt", map_location="cpu")