import itertools
import math
from typing import Tuple, List

def get_n_pairs_with_min_distance(locations: List[Tuple[int, int, int]], n_count:int, spacing: float, min_space: float) -> List[List[Tuple[int, int, int]]]:
  if n_count == 1:
    return list(map(lambda x: [x], locations))
  pairs:  List[List[Tuple[int, int, int]]] = []
  combined = itertools.combinations(locations, n_count)
  rel_dist = (min_space / spacing)
  dist_check = rel_dist * rel_dist
  combined_length = 0
  
  for combination in combined:
    combined_length += 1
    paired = itertools.combinations(combination, 2)
    use = True
    for pair in paired:
      position1, position2 = pair
      width_1, height_1, depth_1 = position1
      width_2, height_2, depth_2 = position2
      delta_width = width_2 - width_1
      delta_height = height_2 - height_1
      delta_depth = depth_2 - depth_1
      dist = delta_width * delta_width + delta_height * delta_height + delta_depth * delta_depth
      
      if dist < dist_check:
        use = False
        break
    if use:
      pairs.append(combination)
      
  print(f'{len(locations)} -> {combined_length}/{len(pairs)}')
  return pairs
  