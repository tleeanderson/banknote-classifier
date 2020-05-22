import csv
import argparse

def read_data(file_path):
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=',')
        return [r for r in reader]

def parse_args():
    parser = argparse.ArgumentParser(description="Classify a banknote as fradulent or authentic")
    parser.add_argument("-i", "--input-file", required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    data = read_data(args.input_file)

    print("len data: {}".format(len(data)))
    print(data[0])
    print(data[-1])
