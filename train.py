import argparse
import importlib
import importlib.util
import os
import sys
import time

import numpy as np
import torch.utils.data
from models.lr_finder import LRFinder

sys.dont_write_bytecode = True
torch.backends.cudnn.benchmark = True  # gotta go fast!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Logger(object):
    """Duplicates all stdout to a file."""

    def __init__(self, path, resume):
        if not resume and os.path.exists(path):
            print(path + " exists")
            sys.exit(0)

        self.log = open(path, "a") if resume else open(path, "w")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        self.stdout.write(message)
        self.stdout.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def import_module(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Train an autoencoder')
    parser.add_argument('experconfig', type=str, help='experiment config file')
    parser.add_argument('--profile', type=str, default="Train", help='config profile')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--lrtest', action='store_true', help='perform learning rate test')
    parser.add_argument('--mpt', action='store_true', help='enable mixed precision training')
    parser.add_argument('--super', action='store_true', help='enable super convergence')

    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()

    outpath = os.path.dirname(args.experconfig)
    log = Logger("{}/log.txt".format(outpath), args.resume)
    print("Python", sys.version)
    print("PyTorch", torch.__version__)
    print(" ".join(sys.argv))
    print("Output path:", outpath)

    # load config
    start_time = time.time()
    experiment_config = import_module(args.experconfig, "config")
    profile = getattr(experiment_config, args.profile)(**{k: v for k, v in vars(args).items() if k not in parsed})
    progress = experiment_config.Progress()
    print("Config loaded ({:.2f} s)".format(time.time() - start_time))

    # load checkpoint
    if args.resume:
        start_time = time.time()
        checkpoint = torch.load("{}/checkpoint.pt".format(outpath))
        iter_num = torch.load("{}/checkpoint.pt".format(outpath))['iteration']
        print("Checkpoint Loaded ({:.2f} s): ".format(time.time() - start_time) +
              "starting at iteration {}".format(iter_num))
    else:
        iter_num = 0

    # build dataset & testing dataset
    start_time = time.time()
    test_dataset = progress.get_dataset()
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=progress.batch_size, shuffle=False)
    test_batch = next(iter(dataloader))

    dataset = profile.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=profile.batch_size, shuffle=True, drop_last=True,
                                             num_workers=16)
    print("Dataset instantiated ({:.2f} s)".format(time.time() - start_time))

    # data writer
    start_time = time.time()
    writer = progress.get_writer()
    print("Writer instantiated ({:.2f} s)".format(time.time() - start_time))

    # build autoencoder
    start_time = time.time()
    ae = profile.get_autoencoder(dataset)
    ae = torch.nn.DataParallel(ae, device_ids=args.devices).to(device).train()
    if args.resume:
        ae.module.load_state_dict(checkpoint['model'], strict=False)
    print("Autoencoder instantiated ({:.2f} s)".format(time.time() - start_time))

    # build optimizer
    start_time = time.time()
    ae_optimizer = profile.get_optimizer(ae.module)
    loss_weights = profile.get_loss_weights()
    if args.resume:
        ae_optimizer.load_state_dict(checkpoint['optimizer'])
    print("Optimizer instantiated ({:.2f} s)".format(time.time() - start_time))

    # build loss function
    start_time = time.time()
    ae_loss = profile.get_loss()
    print("Loss instantiated ({:.2f} s)".format(time.time() - start_time))

    # mixed precision training
    if args.mpt and device == 'cuda':
        from apex import amp
        ae, ae_optimizer = amp.initialize(ae, ae_optimizer, opt_level='O1')

    # perform a learning rate test
    if args.lrtest:
        lr_finder = LRFinder(ae, ae_optimizer, ae_loss, loss_weights, device=device, save_dir=outpath)
        lr_finder.range_test(dataloader, num_iter=1000)
        lr_finder.plot()
        lr_finder.reset()
        print("Max learning rate found: {:.5f}".format(lr_finder.max_lr()))
        sys.exit()

    # build scheduler
    start_time = time.time()
    base_lr = 1e-4
    max_lr = 2e-4 if args.super else base_lr
    scheduler = profile.get_scheduler(ae_optimizer, base_lr, max_lr, iter_num)
    print("Scheduler instantiated ({:.2f} s)".format(time.time() - start_time))

    # train
    start_time = time.time()
    eval_points = np.geomspace(1., profile.max_iter, 100).astype(np.int32)
    prevloss = np.inf

    for epoch in range(1000):
        for data in dataloader:

            # forward
            output = ae(loss_weights.keys(), **{k: x.to(device) for k, x in data.items()})

            # compute final loss
            loss = ae_loss(output, loss_weights)

            # get learning rate from optimizer
            lr = ae_optimizer.param_groups[0]['lr']

            # print current information
            print("Iteration {:06d}: "
                  "lr = {:.5f}, ".format(iter_num, lr) +
                  "loss = {:.5f}, ".format(float(loss.item())) +
                  ", ".join(["{} = {:.5f}".format(
                      k, float(torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v)))
                             for k, v in output["losses"].items()]), end="")

            if iter_num % 10 == 0:
                end_time = time.time()
                ips = 10. / (end_time - start_time)
                print(", iter/sec = {:.2f}".format(ips))
                start_time = time.time()
            else:
                print()

            # compute evaluation output
            if iter_num in eval_points:
                with torch.no_grad():
                    test_output = ae([], **{k: x.to(device) for k, x in test_batch.items()},
                                     **progress.get_ae_args())

                b = data["camera_position"].size(0)
                writer.batch(iter_num, **test_batch, **test_output)

            # update parameters
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()
            if args.super:
                scheduler.step()

            # check for loss explosion
            if loss.item() > 20 * prevloss or not np.isfinite(loss.item()):
                print("Unstable loss function; resetting")

                checkpoint = torch.load("{}/checkpoint.pt".format(outpath))

                ae.module.load_state_dict(checkpoint['model'], strict=False)
                ae_optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler = profile.get_scheduler(ae_optimizer, base_lr, max_lr, checkpoint['iteration'])

            prevloss = loss.item()

            # save intermediate results
            if iter_num % 100 == 0:
                checkpoint = {
                    'iteration': iter_num,
                    'model': ae.module.state_dict(),
                    'optimizer': ae_optimizer.state_dict()
                }
                torch.save(checkpoint, "{}/checkpoint.pt".format(outpath))

            iter_num += 1

        if iter_num >= profile.max_iter:
            break
