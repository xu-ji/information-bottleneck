import os, torch

out_dir = "/network/scratch/x/xu.ji/mutual_info/v21"
data_instances = list(range(3))

archs = [
[1, 128, 1028, 1028, 10], # 1 conv, 1 stoch, 2 det
[1, 64, 512, 512, 10],
[1, 32, 256, 256, 10],
#[1, 16, 128, 128, 10],
]

decays = [0.0, 1e-3] # pick 2

lamb_lrs = [0, 5e-3, 1e-3] # pick 2 + 0 # 5e-4, 1e-4 - assume 1

batch_szs = [128, 32]

seeds = [0, 1]

print(out_dir)

model_ind = 0
for lamb_lr in lamb_lrs:
    for arch in archs:
        for decay in decays:
            for batch_sz in batch_szs:

                # has to be in this order
                for data_instance in data_instances:
                    for seed in seeds:

                        savepath = os.path.join(out_dir, "results_%d.pt" % model_ind)
                        if os.path.exists(savepath):
                            results = torch.load(savepath)
                            model = results["model"]

                            if not hasattr(results["model"], "pre_feats"):
                                print(model_ind)

                        model_ind += 1