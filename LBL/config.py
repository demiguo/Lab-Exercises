import argparse
import datetime

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
parser.add_argument("--adapt_lr_epoch", help="# of epochs to adapt learning rate",
					type=int, default=5)
parser.add_argument("--initial_lr", help="initial learning rate",
					type=float, default=0.001)
parser.add_argument("--dropout", help="Dropout rate",
					type=float, default=0.)
parser.add_argument("--optimizer", help="Optimizer type",
					type=str, default="Adam")

parser.add_argument("--resume", help="Load model instead of training completely new model",
					type=str, default="")
parser.add_argument("--start_epoch", help="Starting epoch number",
					type=int, default=0)

parser.add_argument("--model_dir", help="Models Directory",
					type=str, default="models")

now = datetime.datetime.now()
parser.add_argument('-model_suffix', '--model_suffix', help="Additional Model Information", required=False, default="%s-%s-%s-%s-%s" % (now.year, now.month, now.day, now.hour, now.minute))

args = parser.parse_args()
