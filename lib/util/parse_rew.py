import csv

from typing import Tuple, List
def parse_rew_file(file_path: str, delimiter: str = ',', quotechar: str = '"', comment_char="*") -> Tuple[List[float], List[float]]:
  frequencies: List[float] = []
  values: List[float] = []
  with open(file_path, 'r', encoding="utf-8", newline='') as csv_file:
    csv_without_comments = filter(lambda row: row[0]!=comment_char, csv_file)
    reader = csv.reader(csv_without_comments, delimiter=delimiter, quotechar=quotechar)
    for row in reader:
      frequencies.append(float(row[0]))
      values.append(float(row[1]))

  return (frequencies, values)
