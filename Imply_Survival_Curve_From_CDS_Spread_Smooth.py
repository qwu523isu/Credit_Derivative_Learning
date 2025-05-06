# -*- coding: utf-8 -*-
"""
Created on Tue May  6 11:01:01 2025

@author: Qiong Wu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

# Example market data
maturities = np.array([1, 2, 3, 5])
spreads_bps = np.array([50, 70, 100, 150])
spreads = spreads_bps / 10000.0
R = 0.4

# Initialize arrays
survival_probs = [1.0]
hazard_rates = []
previous_time = 0.0

# Bootstrapping
for i, t in enumerate(maturities):
    delta_t = t - previous_time
    s_i = spreads[i]
    S_prev = survival_probs[-1]
    
    S_current = (1 - R) * S_prev / (s_i * delta_t + (1 - R))
    print(f'The survival probability for maturity {i} equals to {S_current}')
    survival_probs.append(S_current)
    
    h_i = -np.log(S_current / S_prev) / delta_t
    print(f'The default intensity for period {i-1} to {i} equals to {h_i}')
    hazard_rates.append(h_i)
    
    previous_time = t

# Convert lists to arrays
survival_probs = np.array(survival_probs)
hazard_rates = np.array(hazard_rates)

# Display results
print("Maturities (years):", np.insert(maturities, 0, 0.0))
print("Survival Probabilities:", survival_probs)
print("Hazard Rates per interval:")
for i, t in enumerate(maturities):
    print(f"Interval [t_{i}, t_{i+1}]: {hazard_rates[i]:.4f}")

# Original step functions
# Survival curve
times = [0]
survival_steps = [survival_probs[0]]
for i in range(len(maturities)):
    times.extend([maturities[i]])
    survival_steps.extend([survival_probs[i+1]])

# Hazard rate
hazard_times = [0]
hazard_steps = [hazard_rates[0]]
for i in range(len(maturities)):
    if i < len(maturities) - 1:
        hazard_times.extend([maturities[i]])
        hazard_steps.extend([hazard_rates[i]])
    else:
        hazard_times.extend([maturities[i]])
        hazard_steps.extend([hazard_rates[i]])

# Smoothing
# Fine time grid for smooth curves
fine_times = np.linspace(0, max(maturities), 100)

# Hazard rate smoothing
# Define interval start points for hazard rates (t_0 = 0, t_1, ..., t_n)
hazard_points = np.insert(maturities, 0, 0.0)[:-1]
# Linear interpolation
hazard_linear = interp1d(hazard_points, hazard_rates, kind='previous', fill_value='extrapolate')
hazard_smooth_linear = hazard_linear(fine_times)
# Cubic spline interpolation
hazard_cubic = CubicSpline(hazard_points, hazard_rates, bc_type='natural')
hazard_smooth_cubic = hazard_cubic(fine_times)
# Ensure non-negative hazard rates
hazard_smooth_cubic = np.maximum(hazard_smooth_cubic, 0)

# Cumulative hazard: H(t) = ∫ h(u) du
# Using trapezoidal rule for numerical integration
cumulative_hazard_linear = np.cumsum(hazard_smooth_linear * np.diff(np.insert(fine_times, 0, 0)))
cumulative_hazard_cubic = np.cumsum(hazard_smooth_cubic * np.diff(np.insert(fine_times, 0, 0)))

# Smoothed survival: S(t) = exp(-H(t))
survival_smooth_linear = np.exp(-cumulative_hazard_linear)
survival_smooth_cubic = np.exp(-cumulative_hazard_cubic)

# Plotting
# Survival curve
plt.figure(figsize=(10, 6))
plt.step(times, survival_steps, where='post', label='Original Survival Curve', color='blue')
plt.plot(fine_times, survival_smooth_linear, label='Smoothed Survival (Linear)', color='green')
plt.plot(fine_times, survival_smooth_cubic, label='Smoothed Survival (Cubic)', color='red')
plt.xlabel('Time (years)')
plt.ylabel('Survival Probability S(t)')
plt.title('Survival Curve: Original vs. Smoothed')
plt.grid(True)
plt.legend()
plt.savefig('survival_curve_smooth.png')

# Hazard rate
plt.figure(figsize=(10, 6))
plt.step(hazard_times, hazard_steps, where='post', label='Original Hazard Rates', color='blue')
plt.plot(fine_times, hazard_smooth_linear, label='Smoothed Hazard (Linear)', color='green')
plt.plot(fine_times, hazard_smooth_cubic, label='Smoothed Hazard (Cubic)', color='red')
plt.xlabel('Time (years)')
plt.ylabel('Default Intensity h(t)')
plt.title('Default Intensity: Original vs. Smoothed')
plt.grid(True)
plt.legend()
plt.savefig('hazard_rates_smooth.png')