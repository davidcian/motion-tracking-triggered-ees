def hampel_detector(prev_values, current_value, window_size=10, window_offset=-5, k=1.4826, n_accepted_stds=3):
  #print("Prev values", prev_values)
  #print("Current value", current_value)
  all_values = prev_values + [current_value]
  #print("All values", all_values)
  # The index of the latest value
  current_index = len(all_values)
  # The start index of the window
  start_index = max(0, current_index - window_size // 2 + window_offset)
  # The end index of the window
  end_index = min(current_index, current_index + window_size // 2 + window_offset)
  # The values inside the window
  window_values = all_values[start_index:end_index]
  # The median of the window
  window_median = sorted(window_values)[len(window_values) // 2]
  # The absolute deviations inside the window
  window_abs_dev = [abs(win_val - window_median) for win_val in window_values]
  # The median absolute deviation of the window
  window_dev_median = sorted(window_abs_dev)[len(window_abs_dev) // 2]
  # The estimated standard deviation from the MAD
  est_std = k * window_dev_median

  #print("Current value {} Median value {}".format(current_value, window_median))

  return abs(current_value - window_median) > n_accepted_stds * est_std, window_median

def hampel_filter(prev_values, current_value, window_size=10, window_offset=-5, k=1.4826, n_accepted_stds=3, replacement='median'):
  is_anomaly, window_median = hampel_detector(prev_values, current_value, window_size=window_size, window_offset=window_offset, k=k, n_accepted_stds=n_accepted_stds)

  if replacement == 'median':
    replacement_value = window_median
  else:
    replacement_value = 0

  return replacement_value if is_anomaly else current_value

def identity_filter(current_value):
  return current_value

class StatefulHampel():
  def __init__(self):
    self.anomaly_detected = False

  def filter(prev_values, current_value, window_size=10, window_offset=-5, k=1.4826, n_accepted_stds=3, replacement='median'):
    pass
