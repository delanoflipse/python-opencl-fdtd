import csv
import os
import sys
import argparse
from pathlib import Path

if len(sys.argv) == 1:
  print('No file given. Usage: python parse_output.py <path/to/output.csv>')
  sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=Path)

parsed = parser.parse_args()

file_dir = os.path.dirname(__file__)
csv_path = os.path.join(file_dir, parsed.file_path)
print(csv_path)

csv_file = open(csv_path, 'r', encoding="utf-8", newline='')
reader = csv.reader(csv_file, delimiter=',', quotechar='|')

for row in reader:
  print(', '.join(row))

csv_file.close()
