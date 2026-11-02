import os
import argparse
import subprocess
import sys


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', choices=['extrack'], default='extrack', help='training script name')
    parser.add_argument('--config', type=str, default='extrack_b224', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--continual_task', type=str, choices=['day', 'night'], default=None,
                        help='Optional single phase; omit to train Day then Night automatically.')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.mode == "single":
        train_cmd = [sys.executable]
    elif args.mode == "multiple":
        if not args.nproc_per_node or args.nproc_per_node < 1:
            raise ValueError("Set --nproc_per_node to the number of GPUs")
        train_cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                     '--nproc_per_node', str(args.nproc_per_node)]
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_cmd += [os.path.join(root, 'lib', 'train', 'run_training.py'),
                  '--script', args.script, '--config', args.config,
                  '--use_lmdb', str(args.use_lmdb), '--seed', str(args.seed)]
    if args.save_dir:
        train_cmd += ['--save_dir', os.path.abspath(args.save_dir)]
    if args.continual_task:
        train_cmd += ['--continual_task', args.continual_task]
    print(subprocess.list2cmdline(train_cmd))
    subprocess.run(train_cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
