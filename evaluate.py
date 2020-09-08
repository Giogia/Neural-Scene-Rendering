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
    parser.add_argument('profile', type=str, default="Render", help='config profile')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--batchsize', type=int, default=16, help='batchsize')
    parser.add_argument('--image', action='store_true', help='print image of models')
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()

    outpath = os.path.dirname(args.experconfig)
    print("Python", sys.version)
    print("PyTorch", torch.__version__)

    # load config
    start_time = time.time()
    experiment_config = import_module(args.experconfig, "config")
    profile = getattr(experiment_config, args.profile)(**{k: v for k, v in vars(args).items() if k not in parsed})
    progress = experiment_config.Progress()
    print("Config loaded ({:.2f} s)".format(time.time() - start_time))

    # load dataset
    start_time = time.time()
    dataset = profile.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=16)
    print("Dataset instantiated ({:.2f} s)".format(time.time() - start_time))

    # data writer
    start_time = time.time()
    writer = progress.get_writer()
    print("Writer instantiated ({:.2f} s)".format(time.time() - start_time))

    # build autoencoder
    start_time = time.time()
    ae = profile.get_autoencoder(dataset)
    ae = torch.nn.DataParallel(ae, device_ids=args.devices).to(device).eval()
    print("Autoencoder instantiated ({:.2f} s)".format(time.time() - start_time))

    # load
    start_time = time.time()
    checkpoint = torch.load("{}/checkpoint.pt".format(outpath), map_location=torch.device(device))
    state_dict = ae.module.state_dict()
    trained_state_dict = {k: v for k, v in checkpoint['model'].items() if k in state_dict}
    state_dict.update(trained_state_dict)
    ae.module.load_state_dict(state_dict, strict=False)
    print("Training weights loaded ({:.2f} s)".format(time.time() - start_time))

    # eval
    item_num = 0
    iter_num = 0
    start_time = time.time()

    ssim = []
    psnr = []

    with torch.no_grad():
        for data in dataloader:
            batch_size = next(iter(data.values())).size(0)

            # forward
            output = ae([], **{k: x.to(device) for k, x in data.items()}, **profile.get_ae_args())

            if args.image:
                writer.batch(iter_num, ground_truth=False, **output)

            else:
                for k, v in output["metrics"].items():
                    print("{}: {:4f}".format(k, v))

                psnr.append(output['metrics']['psnr'])
                ssim.append(output['metrics']['ssim'])

            end_time = time.time()
            ips = 1. / (end_time - start_time)
            print("{:4} / {:4} ({:.4f} iter/sec)".format(item_num, len(dataset), ips), end="\n")
            start_time = time.time()

            item_num += batch_size
            iter_num += 1

    print("SSIM: {:4f}".format(sum(ssim) / len(ssim)))
    print("PSNR: {:4f}".format(sum(psnr) / len(psnr)))
