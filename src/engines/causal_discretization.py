#!/usr/bin/env python3
"""
Causal Discretization Module
Handles robust data discretization for causal analysis.
"""

import numpy as np

def create_ultra_robust_split_points(series):
    """Create guaranteed monotonically increasing split points with extensive validation"""
    # Remove NaN and infinite values
    clean_series = series.dropna()
    clean_series = clean_series[np.isfinite(clean_series)]

    if len(clean_series) == 0:
        print(f"⚠️ No valid data points for split points, using default [0.0, 1.0]")
        return [0.0, 1.0]

    # Convert to float and get basic stats
    try:
        min_val = float(clean_series.min())
        max_val = float(clean_series.max())
        mean_val = float(clean_series.mean())
        std_val = float(clean_series.std())
    except (ValueError, TypeError) as e:
        print(f"⚠️ Error computing statistics: {e}, using default split points")
        return [0.0, 1.0]

    # Check for invalid values
    if not all(np.isfinite([min_val, max_val, mean_val, std_val])):
        print(f"⚠️ Invalid statistics detected, using default split points")
        return [0.0, 1.0]

    # If there's no variation, create artificial split points
    range_val = max_val - min_val
    if range_val <= 1e-12:  # Essentially no variation
        print(f"⚠️ No variation detected (range={range_val:.2e}), creating artificial splits")
        if abs(min_val) < 1e-6:  # Value is near zero
            return [-1.0, 1.0]
        else:
            margin = max(abs(min_val) * 0.1, 1.0)
            return [min_val - margin, min_val + margin]

    # Calculate minimum separation (at least 1% of range or 1e-6)
    min_separation = max(range_val * 0.01, 1e-6)

    # Strategy 1: Try quantile-based split points
    try:
        unique_values = np.unique(clean_series)
        if len(unique_values) >= 3:
            q33 = float(np.percentile(unique_values, 33.33))
            q67 = float(np.percentile(unique_values, 66.67))

            if q67 - q33 >= min_separation:
                splits = [q33, q67]
                print(f"✅ Using quantile splits: {splits}")
                return splits
    except Exception as e:
        print(f"⚠️ Quantile method failed: {e}")

    # Strategy 2: Try mean ± 0.5 * std
    try:
        if std_val > min_separation / 2:
            split1 = mean_val - 0.5 * std_val
            split2 = mean_val + 0.5 * std_val

            # Ensure splits are within data range
            split1 = max(split1, min_val + min_separation)
            split2 = min(split2, max_val - min_separation)

            if split2 - split1 >= min_separation:
                splits = [float(split1), float(split2)]
                print(f"✅ Using mean±std splits: {splits}")
                return splits
    except Exception as e:
        print(f"⚠️ Mean±std method failed: {e}")

    # Strategy 3: Simple range-based splits
    try:
        split1 = min_val + range_val * 0.4
        split2 = min_val + range_val * 0.6

        # Ensure minimum separation
        if split2 - split1 < min_separation:
            mid_point = (min_val + max_val) / 2
            split1 = mid_point - min_separation / 2
            split2 = mid_point + min_separation / 2

        splits = [float(split1), float(split2)]
        print(f"✅ Using range-based splits: {splits}")
        return splits

    except Exception as e:
        print(f"⚠️ Range-based method failed: {e}")

    # Final fallback: Use data range with padding
    try:
        padding = max(range_val * 0.1, min_separation)
        split1 = min_val + padding
        split2 = max_val - padding

        if split2 <= split1:
            # If still not enough separation, use midpoint approach
            mid_point = (min_val + max_val) / 2
            split1 = mid_point - min_separation / 2
            split2 = mid_point + min_separation / 2

        splits = [float(split1), float(split2)]

        # Final validation - ensure strictly increasing
        if splits[1] <= splits[0]:
            splits[1] = splits[0] + min_separation

        print(f"✅ Using fallback splits: {splits}")
        return splits

    except Exception as e:
        print(f"❌ All methods failed: {e}, using emergency fallback")
        return [0.0, 1.0]
