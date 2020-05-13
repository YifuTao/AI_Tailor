def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="train a model")
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="Total number of epochs for training [25]",
    )
    return parser.parse_args()


import torch
args = parse_args()
device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
a = torch.zeros(10000,200).to(device)
b = int(raw_input('Number of iterations in the neuron network: '))