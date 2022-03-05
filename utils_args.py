import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="Evaluate the performance of the algorithm.")
    parser.add_argument("-f", "--file", help="The dataset to evaluate on.", required=True)
    parser.add_argument("-k", "--k", help="The number of convolutions to use.", default=4, type=int)
    parser.add_argument("-l", "--levels", help="The number of levels to expand to.", default=2, type=int)
    parser.add_argument("-d", "--device", help="The device to run on.", default="cuda:1")
    parser.add_argument("-m", "--model", help="The model to use.", default="allenai/specter")
    parser.add_argument("--mode", help="The mode to use.", default="SGC")
    parser.add_argument("--max_examples", type=int, help="The maximum number of examples to use.", default=100)
    parser.add_argument(
        "--expansion_embeddings_path",
        help="The expansion embeddings to use.",
        default="sciad_data/expansion_embeddings_normalized_specter_cls.npy",
    )
    return parser
