import argparse
import importlib
import importlib.util
import os
import sys
import time

import torch.utils.data

sys.dont_write_bytecode = True

torch.backends.cudnn.benchmark = True  # gotta go fast!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def import_module(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Render')
    parser.add_argument('experconfig', type=str, help='experiment config')
    parser.add_argument('profile', type=str, help='config profile')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--batchsize', type=int, default=16, help='batchsize')
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()

    outpath = os.path.dirname(args.experconfig)
    print(" ".join(sys.argv))
    print("Output path:", outpath)

    # load config
    experconfig = import_module(args.experconfig, "config")
    profile = getattr(experconfig, args.profile)(**{k: v for k, v in vars(args).items() if k not in parsed})

    # load dataset
    dataset = profile.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=16)

    # data writer
    writer = profile.get_writer()

    # build autoencoder
    ae = profile.get_autoencoder(dataset)
    ae = torch.nn.DataParallel(ae, device_ids=args.devices).to(device).eval()

    # load
    state_dict = ae.module.state_dict()
    trained_state_dict = torch.load("{}/aeparams.pt".format(outpath), map_location=torch.device(device))
    trained_state_dict = {k: v for k, v in trained_state_dict.items() if k in state_dict}
    state_dict.update(trained_state_dict)
    ae.module.load_state_dict(state_dict, strict=False)

    # eval
    item_num = 0
    start_time = time.time()

    with torch.no_grad():
        for data in dataloader:
            b = next(iter(data.values())).size(0)

            # forward
            output = ae([], **{k: x.to(device) for k, x in data.items()}, **profile.get_ae_args())

            writer.batch(item_num + torch.arange(b), **data, **output)

            end_time = time.time()
            ips = 1. / (end_time - start_time)
            print("{:4} / {:4} ({:.4f} iter/sec)".format(item_num, len(dataset), ips), end="\n")
            start_time = time.time()

            item_num += b

    # cleanup
    writer.finalize()
