# EXP.NO.11-Simulation-of-Spread-Spectrum-Modulation-Techniques


# AIM:
To simulate the process of Direct Sequence Spread Spectrum (DSSS) modulation using Binary Phase Shift Keying (BPSK).

# SOFTWARE REQUIRED:
Python (Version 3.x)
NumPy (for numerical operations)
Matplotlib (for plotting the graphs)

# ALGORITHM:
1. Generate random binary data.
2. Create a PN sequence of length 8.
3. Perform BPSK modulation on the data (0 → -1, 1 → +1).
4. Spread the BPSK modulated signal using the PN sequence.
5. Modulate the spread signal using a BPSK carrier.
6. Plot the DSSS spread signal and the BPSK modulated carrier waveform.

# PROGRAM:
```python
import numpy as np
import matplotlib.pyplot as plt

# System Parameters
data_length = 4                # Number of bits
chips_per_bit = 8              # PN sequence length
bit_rate = 1e3                 # 1 kbps
chip_rate = bit_rate * chips_per_bit
carrier_freq = 20e3            # 20 kHz carrier
sample_rate = 160e3            # 160 kHz sampling rate
samples_per_chip = int(sample_rate / chip_rate)

# Generate random binary data
def generate_data(length):
    return np.random.randint(0, 2, length)

import numpy as np
import matplotlib.pyplot as plt

# System Parameters
data_length = 4                # Number of bits
chips_per_bit = 8              # PN sequence length
bit_rate = 1e3                 # 1 kbps
chip_rate = bit_rate * chips_per_bit
carrier_freq = 20e3            # 20 kHz carrier
sample_rate = 160e3            # 160 kHz sampling rate
samples_per_chip = int(sample_rate / chip_rate)

# Generate random binary data
def generate_data(length):
    return np.random.randint(0, 2, length)

# Generate PN sequence: ±1 chips
def generate_pn_sequence(length):
    return np.random.choice([-1, 1], length)

# BPSK mapping: 0 → -1, 1 → +1
def bpsk_modulate(bit):
    return 2 * bit - 1

# DSSS spreading
def dsss_spread(data, pn_sequence):
    spread = []
    for bit in data:
        bpsk_bit = bpsk_modulate(bit)
        spread.extend(bpsk_bit * pn_sequence)
    return np.array(spread)

# BPSK carrier modulation of spread signal
def carrier_modulate(spread_signal, carrier_freq, sample_rate, samples_per_chip):
    total_samples = len(spread_signal) * samples_per_chip
    t = np.arange(total_samples) / sample_rate
    carrier_wave = np.cos(2 * np.pi * carrier_freq * t)

    # Repeat each chip to match carrier sampling
    chip_samples = np.repeat(spread_signal, samples_per_chip)
    return chip_samples * carrier_wave, t

# Main function
if __name__ == "__main__":
    # Generate input
    data = generate_data(data_length)
    pn_seq = generate_pn_sequence(chips_per_bit)

    print("Original Data Bits:     ", data)
    print("PN Sequence:            ", pn_seq)
    
    # DSSS spreading
    spread_signal = dsss_spread(data, pn_seq)

    # BPSK Carrier modulation
    bpsk_waveform, t = carrier_modulate(spread_signal, carrier_freq, sample_rate, samples_per_chip)

    # Plot DSSS spread signal (chip values)
    plt.figure(figsize=(12, 3))
    plt.plot(spread_signal, drawstyle='steps-mid')
    plt.title("DSSS Spread Signal (Baseband)")
    plt.xlabel("Chip Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()

    # Plot BPSK modulated carrier waveform
    plt.figure(figsize=(12, 3))
    plt.plot(t, bpsk_waveform)
    plt.title("BPSK Modulated Waveform (Carrier)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```

# OUTPUT:
![image](https://github.com/user-attachments/assets/a2a86ba7-ae27-4627-83c2-2429a511e766)
![image](https://github.com/user-attachments/assets/b112b19b-2a76-4beb-9b8f-07c386d3cc76)

# RESULT:
The DSSS spread signal is displayed as a plot showing the baseband spread signal after applying the PN sequence. The BPSK modulated signal is shown as a plot of the BPSK modulated carrier waveform.
