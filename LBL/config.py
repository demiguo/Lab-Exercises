import argparse

parser = argparse.ArgumentParser()

# required arguments
parser.add_argument("--context_size", help="number of words taken into account in context",
					type=int, default=5)
parser.add_argument("--epochs", help="number of epochs",
					type=int, default=20)
parser.add_argument("--batch_size", help="batching size",
					type=int, default=128)
parser.add_argument("--GPU", help="use GPU optimizer (1 to use GPU, 0 for no GPU)",
					type=int, default=0)
parser.add_argument("--mode", help="specify test, validate, or train",
					type=str, default="test")
parser.add_argument("--lr", help="learning rate",
					type=float, default=0.001)

args = parser.parse_args()
