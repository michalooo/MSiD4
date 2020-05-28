import time

from nn.models import get_model
import train

x = 10


def validate(model, model_path, split_file):
  m = get_model(model)
  print("Loading weights from %s" % model_path)
  m.load_variables(model_path)
  x_train, y_train, x_val, y_val = train.load_data(False, True, False, split_file)

  x_val = x_val[:2500]

  print(x_val.shape)
  y_val = y_val[:2500]

  print(y_val.shape)
  return train.validate(m, x_val, y_val)


def parse_args():
  import argparse
  parser = argparse.ArgumentParser(description="Validate neural net")
  parser.add_argument("-m", "--model", default="SimpleNet")
  parser.add_argument("--split-file", type=str, default="train_val_split1.npz")
  parser.add_argument("model_path")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  time1 = time.time()
  accuracy = validate(args.model, args.model_path, args.split_file)
  print("Validation accuracy: %f" % accuracy)
  time2 = time.time()
  print(time2 - time1)