# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 22:54:21 2025

@author: Qiong Wu
"""
# Clear all variables in the current Python environment
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

# Confirm variables are cleared
print("All local variables have been cleared.")

# Clear console
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_console()
"-----------------------------------------------------------------------------"
"-----------------------------------------------------------------------------"
"-----------------------------------------------------------------------------"

import numpy as np
import matplotlib.pyplot as plt

# Example market data
# Maturities (years) and the corresponding CDS spreads in basis points (bps)
# (Note: 1 bp = 0.0001 in decimal terms)
maturities = np.array([1, 2, 3, 5])
spreads_bps = np.array([50, 70, 100, 150])
# Convert CDS spreads from basis points to decimal
spreads = spreads_bps / 10000.0

# Recovery rate (for example, 40% recovery implies R = 0.4)
R = 0.4

# Initialize arrays to store survival probabilities and hazard rates
# S(0) is always 1 (no default at time 0)
survival_probs = [1.0]
hazard_rates = []

# Bootstrapping: for each interval, solve for S(t_i) and then h_i
previous_time = 0.0
for i, t in enumerate(maturities):
    delta_t = t - previous_time
    s_i = spreads[i]
    S_prev = survival_probs[-1]
    
    # The relationship between the premium and protection leg in the interval is:
    #   s_i * delta_t * S(t) = (1 - R) * (S_prev - S(t))
    #
    # We solve for S(t) (denoted here as S_current):
    #   s_i * delta_t * S_current + (1 - R) * S_current = (1 - R) * S_prev
    #   S_current * [s_i * delta_t + (1 - R)] = (1 - R) * S_prev
    S_current = (1 - R) * S_prev / (s_i * delta_t + (1 - R))
    print(f'The survival probability for maturity {i} equals to {S_current}')
    survival_probs.append(S_current)
    
    # Given the survival probability in the interval, we solve for h_i where:
    #   S_current = S_prev * exp(-h_i * delta_t)
    # => h_i = -1/delta_t * ln(S_current / S_prev)
    h_i = -np.log(S_current / S_prev) / delta_t
    print(f'The default intensity for period {i-1} to {i} equals to {h_i}')
    hazard_rates.append(h_i)
    
    previous_time = t

# Convert lists to NumPy arrays for consistency
survival_probs = np.array(survival_probs)
hazard_rates = np.array(hazard_rates)

# Display the results
print("Maturities (years):", np.insert(maturities, 0, 0.0))
print("Survival Probabilities:", survival_probs)
print("Hazard Rates per interval:")
for i, t in enumerate(maturities):
    print(f"Interval [t_{i}, t_{i+1}]: {hazard_rates[i]:.4f}")

# Visualization: plot the survival curve
# For plotting purposes, we'll create a step function that holds the survival probability constant within each interval.

# Given survival_probs and maturities arrays
# Example (using the computed arrays):
# survival_probs = [S(0), S(1), S(2), S(3), S(5)]
# maturities = [1, 2, 3, 5]

# Survival curve plot
times = [0]
survival_steps = [survival_probs[0]]
# Loop over the number of intervals (which equals len(maturities))
for i in range(len(maturities)):
    # Use maturities[i] for the end time of the interval
    times.extend([maturities[i]])
    survival_steps.extend([survival_probs[i+1]])

# For a step plot, we want the survival probability constant over each interval.
plt.figure(figsize=(8, 5))
plt.step(times, survival_steps, where='post')
plt.xlabel('Time (years)')
plt.ylabel('Survival Probability S(t)')
plt.title('Bootstrapped Survival Curve')
plt.grid(True)
plt.show()

# Hazard rate plot
hazard_times = [0]
hazard_steps = [hazard_rates[0]]
for i in range(len(maturities)):
    if i < len(maturities) - 1:
        hazard_times.extend([maturities[i]])
        hazard_steps.extend([hazard_rates[i]])
    else:
        hazard_times.extend([maturities[i]])
        hazard_steps.extend([hazard_rates[i]])

plt.figure(figsize=(8, 5))
plt.step(hazard_times, hazard_steps, where='post')
plt.xlabel('Time (years)')
plt.ylabel('Default Intensity h(t)')
plt.title('Bootstrapped Default Intensity')
plt.grid(True)
plt.savefig('hazard_rates.png')