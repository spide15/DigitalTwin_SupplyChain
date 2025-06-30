"""
Helper functions for simulating traffic congestion.
"""
import numpy as np

def simulate_traffic(zone, base_speed):
    """Return speed multiplier based on simulated congestion."""
    congestion = np.random.uniform(0.7, 1.3)  # 0.7x to 1.3x speed
    return base_speed * congestion
