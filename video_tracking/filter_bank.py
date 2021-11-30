def hampel_filter(prev_values, current_value, window_size=10, window_offset=-5, k=1.4826, n_accepted_stds=3):
  all_values = prev_values + current_value
  # The index of the latest value
  current_index = len(all_values)
  # The start index of the window
  start_index = max(0, current_index - window_size // 2 + window_offset)
  # The end index of the window
  end_index = min(current_index, current_index + window_size // 2 + window_offset)
  # The values inside the window
  window_values = all_values[start_index, end_index]
  # The median of the window
  window_median = window_values[len(window_values) // 2]
  # The absolute deviations inside the window
  window_abs_dev = [abs(win_val - window_median) for win_val in window_values]
  # The median absolute deviation of the window
  window_dev_median = [len(window_abs_dev) // 2]
  # The estimated standard deviation from the MAD
  est_std = k * window_dev_median

  return window_median if abs(current_value - window_median) > n_accepted_stds * est_std else current_value
