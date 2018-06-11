from neurotorch.core.Trainer import Trainer
import argparse
import importlib


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a neural network model')
    parser.add_argument('NET', help="Neural network")
    parser.add_argument('INPUTS', help="Input dataset for training")
    parser.add_argument('LABELS', help="Label dataset for training")
    parser.add_argument('-c', '--checkpoint', help="Checkpoint number",
                        type=int)
    parser.add_argument('-d', '--gpu', help="GPU number for training",
                        type=int)

    return parser.parse_args()


def main():
    args = parse_arguments()
    net_path = "neurotorch.nets." + args.NET
    importlib.import_module(net_path)

    if args.checkpoint:
        trainer = Trainer(net_path, args.DATASET, checkpoint=args.checkpoint)

    else:
        trainer = Trainer(net_path, args.DATASET)

    trainer.start_training()


if __name__ == '__main__':
    main()
