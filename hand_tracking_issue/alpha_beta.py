import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_live_signal(t):
    return np.sin(t) + np.random.normal(0, 0.1)

velocity = 0
filtered_signal = 0
delta_t = 0.01
alpha = 0.3
beta = 0.1
def alpha_beta_filter(signal, alpha, beta):
    # Initial estimates
    global filtered_signal, velocity
    predicted_signal = filtered_signal + velocity * delta_t   
    filtered_signal = predicted_signal + alpha * (signal - predicted_signal)
    velocity = velocity + beta * ((signal - predicted_signal) / delta_t)
    return filtered_signal

# Parameters
t = 0
max_display_time = 10  # Maximum time to display on the screen
t_data = []
signal_data = []
filtered_data = []

# Set up plot
fig, axs = plt.subplots(1)
line1, = axs[0].plot([], [], 'b-', label='Noisy Signal', lw = 2)
line2, = axs[1].plot([], [], 'r-', label='Filtered Signal', lw = 2)
axs[0].legend()
axs[1].legend()

# Function to update plot
def update(frame):
    global t, t_data, signal_data, filtered_data
    
    # Generate live signal
    live_signal = generate_live_signal(t)
    
    # Apply alpha-beta filter
    if t == 0:
        filtered_signal = live_signal
    else:
        filtered_signal = alpha_beta_filter(live_signal, alpha, beta)
    
    # Update data
    if t < max_display_time:
        t_data.append(t)
        signal_data.append(live_signal)
        filtered_data.append(filtered_signal)
    else:
        t_data[:-1] = t_data[1:]  # Shift the array
        t_data[-1] = t
        signal_data[:-1] = signal_data[1:]  # Shift the array
        signal_data[-1] = live_signal
        filtered_data[:-1] = filtered_data[1:]  # Shift the array
        filtered_data[-1] = filtered_signal
    
    # Update plot
    line1.set_data(t_data, signal_data)
    line2.set_data(t_data, filtered_data)
    
    a = t_data[0]
    b = t_data[-1]
    # Update x-axis limits
    axs[0].set_xlim(a, b)
    axs[1].set_xlim(a, b)
    print(t_data[0]," ", t_data[-1])
    
    # Update y-axis limits
    min_val = min(min(signal_data), min(filtered_data))
    max_val = max(max(signal_data), max(filtered_data))
    axs.set_ylim(min_val - 0.5, max_val + 0.5)
    axs[1].set_ylim(min_val - 0.5, max_val + 0.5)
    
    t += 0.1
    return (line1, line2)

# Animate plot
ani = FuncAnimation(fig, update, frames=1000, interval=200, repeat=False)
plt.title('Alpha-Beta Filter Live Demo')
plt.grid(True)
plt.plot()
plt.show()
