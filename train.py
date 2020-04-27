# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Train an autoencoder."""
import argparse
import importlib
import importlib.util
import os
import re
import sys
import time

import numpy as np
import torch.utils.data
from apex import amp
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

        iternum = 0
        if resume:
            with open(path, "r") as f:
                for line in f.readlines():
                    match = re.search("Iteration (\d+).* ", line)
                    if match is not None:
                        it = int(match.group(1))
                        if it > iternum:
                            iternum = it
        self.iternum = iternum

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
    experconfig = import_module(args.experconfig, "config")
    profile = getattr(experconfig, args.profile)(**{k: v for k, v in vars(args).items() if k not in parsed})
    progressprof = experconfig.Progress()
    print("Config loaded ({:.2f} s)".format(time.time() - start_time))

    # build dataset & testing dataset
    start_time = time.time()
    testdataset = progressprof.get_dataset()
    dataloader = torch.utils.data.DataLoader(testdataset, batch_size=progressprof.batchsize, shuffle=False,
                                             drop_last=True, num_workers=0)
    for testbatch in dataloader:
        break
    dataset = profile.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=profile.batchsize, shuffle=True, drop_last=True,
                                             num_workers=16)
    print("Dataset instantiated ({:.2f} s)".format(time.time() - start_time))

    # data writer
    start_time = time.time()
    writer = progressprof.get_writer()
    print("Writer instantiated ({:.2f} s)".format(time.time() - start_time))

    # build autoencoder
    start_time = time.time()
    ae = profile.get_autoencoder(dataset)
    ae = torch.nn.DataParallel(ae, device_ids=args.devices).to(device).train()
    if args.resume:
        ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
    print("Autoencoder instantiated ({:.2f} s)".format(time.time() - start_time))

    # build optimizer
    start_time = time.time()
    ae_optimizer = profile.get_optimizer(ae.module)
    loss_weights = profile.get_loss_weights()
    print("Optimizer instantiated ({:.2f} s)".format(time.time() - start_time))

    # build loss function
    start_time = time.time()
    ae_loss = profile.get_loss()
    print("Loss instantiated ({:.2f} s)".format(time.time() - start_time))

    # GPU optimization - mixed precision training
    if args.mpt and device == 'cuda':
        ae, ae_optimizer = amp.initialize(ae, ae_optimizer, opt_level='O1')

    # super convergence
    if args.lrtest:
        lr_finder = LRFinder(ae, ae_optimizer, ae_loss, loss_weights, device=device, save_dir=outpath)
        lr_finder.range_test(dataloader, end_lr=0.05, num_iter=30)
        lr_finder.plot()
        lr_finder.reset()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(ae_optimizer,
                                                    max_lr=4e-3, epochs=1000, steps_per_epoch=len(dataloader))

    # train
    start_time = time.time()
    evalpoints = np.geomspace(1., profile.maxiter, 100).astype(np.int32)
    iternum = log.iternum
    prevloss = np.inf

    for epoch in range(1000):
        for data in dataloader:

            # forward
            output = ae(loss_weights.keys(), **{k: x.to(device) for k, x in data.items()})

            # compute final loss
            loss = ae_loss(output, loss_weights)

            print('LOSS:', loss.item(), 'PREVLOSS:', prevloss)

            # print current information
            print("Iteration {}: loss = {:.5f}, ".format(iternum, float(loss.item())) +
                  ", ".join(["{} = {:.5f}".format(k,
                                                  float(torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v,
                                                                                                        tuple) else torch.mean(
                                                      v)))
                             for k, v in output["losses"].items()]), end="")
            if iternum % 10 == 0:
                endtime = time.time()
                ips = 10. / (endtime - start_time)
                print(", iter/sec = {:.2f}".format(ips))
                start_time = time.time()
            else:
                print()

            # compute evaluation output
            if iternum in evalpoints:
                with torch.no_grad():
                    testoutput = ae([], **{k: x.to(device) for k, x in testbatch.items()},
                                    **progressprof.get_ae_args())

                b = data["campos"].size(0)
                writer.batch(iternum, iternum * profile.batchsize + torch.arange(b), **testbatch, **testoutput)

            # update parameters
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()
            scheduler.step()

            # check for loss explosion
            if loss.item() > 20 * prevloss or not np.isfinite(loss.item()):
                print("Unstable loss function; resetting")

                ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
                ae_optimizer = profile.get_optimizer(ae.module)

            prevloss = loss.item()

            # save intermediate results
            if iternum % 10 == 0:
                torch.save(ae.module.state_dict(), "{}/aeparams.pt".format(outpath))

            iternum += 1

        if iternum >= profile.maxiter:
            break

    # cleanup
    writer.finalize()
