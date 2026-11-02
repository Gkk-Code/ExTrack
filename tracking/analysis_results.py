import argparse
import _init_paths
from lib.test.evaluation import get_dataset, trackerlist
from lib.test.analysis.plot_results import print_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate ExTrack tracking results')
    parser.add_argument('--dataset_name', '--dataset', required=True)
    parser.add_argument('--config', default='extrack_b224')
    args = parser.parse_args()
    trackers = trackerlist('extrack', args.config, args.dataset_name, display_name='ExTrack')
    print_results(trackers, get_dataset(args.dataset_name), args.dataset_name,
                  merge_results=True, plot_types=('success', 'prec', 'norm_prec'))


if __name__ == '__main__':
    main()
