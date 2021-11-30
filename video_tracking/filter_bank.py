def filter_z(depth_values, current_z):
  max_z_deviation = 0.2
  if current_z > depth_values[-1] + max_z_deviation or current_z < depth_values[-1] - max_z_deviation:
    filtered_z = depth_values[-1]
  else:
    filtered_z = current_z

  return filtered_z