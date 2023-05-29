import os
import torch
import traceback

# printing train, test statistics for accepted models

# from main_swag

for out_dir in ["/network/scratch/x/xu.ji/mutual_info/v24"]:

    for do_lambs in [[0]]:
        for do_batches in [[128, 32]]:

            print((out_dir, do_lambs, do_batches))

            data_instances = list(range(3))

            archs = [
            [1, 16, 128, 128, 10], #[1, 128, 1028, 1028, 10], # actually smaller one arch_0
            [1, 64, 512, 512, 10],
            [1, 32, 256, 256, 10],
            ]

            decays = [0.0, 1e-3] # pick 2

            lamb_lrs = [0, 1e-3]

            batch_szs = [128, 32]

            seeds = [0, 1]

            train_accs = []
            test_accs = []
            train_losses = []
            test_losses = []

            thresh = 0.0 # todo
            skipped = 0
            skipped_file = 0
            considered = 0

            neg_gen_gap = 0

            l_MIs = []


            print("Doing lamb lr %s batch %s" % (do_lambs, do_batches))

            model_ind = 0
            for lamb_lr in lamb_lrs:
                for arch in archs:
                    for decay in decays:
                        for batch_sz in batch_szs:

                            for data_instance in data_instances:
                                for seed in seeds:

                                    try:
                                        r = torch.load(os.path.join(out_dir, "results_%d.pt" % model_ind))

                                        if lamb_lr in do_lambs and batch_sz in do_batches:
                                            if (r["train_acc"] > thresh):
                                                train_accs.append(r["train_acc"])
                                                test_accs.append(r["test_acc"])

                                                train_losses.append(r["train_loss"])
                                                test_losses.append(r["test_loss"])

                                                l_MIs.append(r["diagnostics"]["l_MIs"][-1])
                                            else:
                                                skipped += 1

                                            if r["test_loss"] < r["train_loss"]:
                                                neg_gen_gap += 1

                                            considered += 1

                                    except:
                                        #print("Skipping %s, batch_sz %s" % (model_ind, batch_sz))
                                        #print(traceback.format_exc())
                                        skipped_file += 1

                                    model_ind += 1

            train_accs = torch.tensor(train_accs)
            test_accs = torch.tensor(test_accs)
            train_losses = torch.tensor(train_losses)
            test_losses = torch.tensor(test_losses)

            l_MIs = torch.tensor(l_MIs)

            print("Num neg gen gap")
            print(neg_gen_gap)

            print("All:")
            print(model_ind)

            print("Considered:")
            print(considered)

            print("Skipped:")
            print(skipped)

            print("Skipped_file:")
            print(skipped_file)

            print("Resulting:")
            print(train_accs.shape)

            print("mean, std, max, min")
            print("Train loss & %.4f & %.4f & %.4f & %.4f \\\\" % (train_losses.mean(), train_losses.std(), train_losses.max(), train_losses.min() ))
            print("Train accuracy & %.4f & %.4f & %.4f & %.4f \\\\" % (train_accs.mean(), train_accs.std(), train_accs.max(), train_accs.min()))

            print("Test loss & %.4f & %.4f & %.4f & %.4f \\\\" % ( test_losses.mean(), test_losses.std(), test_losses.max(), test_losses.min() ))
            print("Test accuracy & %.4f & %.4f & %.4f & %.4f \\\\" % (test_accs.mean(), test_accs.std() , test_accs.max(), test_accs.min()))


            print("l MIs & %.4f & %.4f & %.4f & %.4f \\\\" % (l_MIs.mean(), l_MIs.std(), l_MIs.max(), l_MIs.min() ))

            print("")
