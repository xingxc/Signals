# %%
# %% adding some git changes
import time

import matplotlib.pyplot as plt
import numpy as np

import fourier_transform as ft


# %% adding some changes here

N = 1000
win0 = ft.windows(100)
win1 = ft.windows(200)
win2 = ft.windows(300)
time = np.arange(0, N, 1),

win_hanning0 = ft.fourier_transform(time, win0.window_dict['hanning'])
win_hanning1 = ft.fourier_transform(time, win1.window_dict['hanning'])
win_hanning2 = ft.fourier_transform(time, win2.window_dict['hanning'])
win_hanning0.FFT_FFT(N=20*N)
win_hanning1.FFT_FFT(N=20*N)
win_hanning2.FFT_FFT(N=20*N)

scale = N*100
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.plot(win_hanning0.signal_bins, win_hanning0.signal_dB, label='100')
ax.plot(win_hanning1.signal_bins, win_hanning1.signal_dB, label='200')
ax.plot(win_hanning2.signal_bins, win_hanning2.signal_dB, label='300')
xlim = 100
xcenter = 0
# ax.set_xlim([xcenter-xlim, xcenter+xlim])
ax.set_ylim([-200, 0])
ax.grid(linestyle='--')
ax.set_ylabel('dB')
ax.set_xlabel('Frequency')
ax.legend()
plt.show()
# %%


# def cft(g, f):
#     """Numerically evaluate the Fourier Transform of g for the given frequencies"""
#     result = np.zeros(len(f), dtype=complex)

#    # Loop over all frequencies and calculate integral value
#     for i, ff in enumerate(f):
#         # Evaluate the Fourier Integral for a single frequency ff,
#         # assuming the function is time-limited to abs(t)<5
#         result[i] = complex_quad(lambda t: g(t)*np.exp(-2j*np.pi*ff*t), -5, 5)
#     return result


# def complex_quad(g, a, b):
#     """Return definite integral of complex-valued g from a to b,
#     using Simpson's rule"""
#     # 2501: Amount of used samples for the trapezoidal rule
#     t = np.linspace(a, b, 2501)
#     x = g(t)
#     return integrate.simps(y=x, x=t)


# # %% Window Analysis
# # time, signal_Y = ft.create_signal(
# #     1024, 0, 10, amplitude=[1], signal_freq=[1024], signal_phase=[0], signal_type='cos')
# time, signal_Y = ft.create_signal(
#     1000, 0, 1, [1], [100], [0], 'cos')

# N = time.__len__()
# win = ft.windows(N)

# win_rect = ft.fourier_transform(time, win.window_dict['rect'])
# win_hanning = ft.fourier_transform(time, win.window_dict['hanning'])
# win_rect.FFT_FFT(N=100*N)
# win_hanning.FFT_FFT(N=100*N)

# fig, ax = plt.subplots(1, 1, figsize=(15, 15))
# ax.plot(win_rect.signal_bins, win_rect.signal_dB, label='Rect')
# ax.plot(win_hanning.signal_bins, win_hanning.signal_dB, label='Hanning')

# xlim = 300
# xcenter = 0
# xcenter = win_rect.signal_bins[abs(win_hanning.signal_freq - xcenter).argmin()]
# ax.set_xlim([xcenter-xlim, xcenter+xlim])
# ax.set_ylim([-100, 1])

# ax.grid(linestyle='--')
# ax.set_xlabel('Bins')
# ax.set_ylabel('dB')
# ax.legend()
# # %%
# time, signal_Y = ft.create_signal(
#     1000, 0, 1, [1], [1], [0], 'cos')

# N = time.__len__()
# win = ft.windows(N)

# win_rect = ft.fourier_transform(time, win.window_dict['rect'])
# win_hanning = ft.fourier_transform(time, win.window_dict['hanning'])
# win_hamming = ft.fourier_transform(time, win.window_dict['hamming'])
# win_blackman = ft.fourier_transform(time, win.window_dict['blackman'])
# win_kaiser = ft.fourier_transform(time, win.window_dict['kaiser'])

# win_rect.FFT_FFT(N=10*N)
# win_hanning.FFT_FFT(N=10*N)
# win_hamming.FFT_FFT(N=3*N)
# win_blackman.FFT_FFT(N=3*N)
# win_kaiser.FFT_FFT(N=3*N)

# # Plot windows
# fig, ax = plt.subplots(1, 1, figsize=(15, 15))
# ax.plot(win_rect.signal_bins, win_rect.signal_dB, label='Rect')
# ax.plot(win_hanning.signal_bins, win_hanning.signal_dB, label='Hanning')

# ax.plot(win_rect.signal_freq, win_rect.signal_dB, label='Rect')
# ax.plot(win_hanning.signal_freq, win_hanning.signal_dB, label='Hanning')
# ax.plot(win_hamming.signal_bins, win_hamming.signal_dB, label='Hamming')
# ax.plot(win_blackman.signal_bins, win_blackman.signal_dB, label='Blackman')
# ax.plot(win_kaiser.signal_bins, win_kaiser.signal_dB, label='Kaiser')
# ax.set_xlim([-25, 25])
# # ax.set_xticks(np.arange(-25, 25, 1))

# # ax.set_ylim([-100, 5])
# ax.grid(linestyle='--')
# ax.set_xlabel('Bins')
# ax.set_ylabel('dB')
# ax.legend()

# # ax[1].set_xlim([-10, 10])
# # ax[1].set_ylim([-100, 5])
# # ax[1].grid(linestyle='--')
# # ax[1].set_xlabel('Bins')
# # ax[1].set_ylabel('dB')
# # ax[1].legend()


# # %%
# N = 1000
# win = ft.windows(N)

# time = np.arange(0, N, 1)
# win_hanning = ft.fourier_transform(time, win.window_dict['hanning'])
# win_hanning.FFT_FFT(N=N*4)


# def response(f):
#     a_real = 0
#     a_imag = 0

#     for j in range(N):
#         a_real += win.window_dict['hanning'][j] * \
#             np.cos(2 * math.pi * f * j / N)
#         a_imag += win.window_dict['hanning'][j] * \
#             np.sin(2 * math.pi * f * j / N)

#     S1 = win.window_dict['hanning'].sum()

#     return np.sqrt(a_real**2 + complex(a_imag)**2)/S1


# f = np.linspace(0, 100, 101)
# output = []
# for item in f:
#     output.append(response(item))

# output = np.array(output)


# plt.plot(f, output)

# # %% Sinusoidal Analysis
# signal_X, signal_Y = ft.create_signal(
#     1024, 0, 2, [1], [10], [math.pi/6], 'cos')
# obj_ref = ft.fourier_transform(signal_X, signal_Y)
# obj_user = ft.fourier_transform(signal_X, signal_Y)

# obj_ref.FFT_FFT(solver='scipy', norm=None)
# obj_user.FFT_FFT(N=2048*4, solver='scipy', norm=None)
# obj_ref.signal_threshold(10000, write=True)
# obj_user.signal_threshold(10000, write=True)


# # %% Plot Sinusoidals
# fig, ax = plt.subplots(6, 1, figsize=(20, 60))

# ax[0].plot(obj_ref.signal_freq, obj_ref.signal_dB, marker='o',
#            linestyle='-',  color='red', label='ref (N = 2048)')

# ax[0].plot(obj_user.signal_freq, obj_user.signal_dB,
#            linestyle='--',  color='blue', label='user (N = ??)')

# ax[1].plot(obj_ref.signal_freq, obj_ref.signal_dB_thr, marker='v',
#            linestyle='-',  color='red', label='ref (N = 2048)')

# ax[1].plot(obj_user.signal_freq, obj_user.signal_dB_thr,
#            linestyle='--',  color='blue', label='user (N = ??)')


# ax[2].plot(obj_ref.signal_freq, obj_ref.signal_phase,
#            linestyle='-',  color='red', label='ref (N = 2048)')

# ax[2].plot(obj_user.signal_freq, obj_user.signal_phase,
#            linestyle='--',  color='blue', label='user (N = ??)')


# ax[3].plot(obj_ref.signal_freq, obj_ref.signal_phase_thr,
#            linestyle='-',  color='red', label='ref (N = 2048)')

# ax[3].plot(obj_user.signal_freq, obj_user.signal_phase_thr,
#            linestyle='--',  color='blue', label='user (N = ??)')


# ax[4].plot(obj_ref.signal_freq, obj_ref.signal_F.imag, marker='v',
#            linestyle='-',  color='red', label='ref (N = 2048)')

# ax[4].plot(obj_user.signal_freq, obj_user.signal_F.imag,
#            linestyle='--',  color='blue', label='user (N = ??)')

# ax[5].plot(obj_ref.signal_freq, obj_ref.signal_F_thr.imag, marker='v',
#            linestyle='-',  color='red', label='ref (N = 2048)')

# ax[5].plot(obj_user.signal_freq, obj_user.signal_F_thr.imag,
#            linestyle='--',  color='blue', label='user (N = ??)')

# view_center = 25
# ax[0].set_title('signal dB')
# ax[1].set_title('signal dB (thr)')
# ax[2].set_title('signal phase')
# ax[3].set_title('signal phase (thr)')
# ax[4].set_title('signal imaginary')
# ax[5].set_title('signal imaginary (thr)')

# ax[0].set_xlim(-view_center, view_center)
# ax[1].set_xlim(-view_center, view_center)
# ax[2].set_xlim(-view_center, view_center)
# ax[3].set_xlim(-view_center, view_center)
# ax[4].set_xlim(-view_center, view_center)
# ax[5].set_xlim(-view_center, view_center)

# ax[0].set_ylim(-100, 0)
# ax[1].set_ylim(-40, 0)

# # ax[4].set_ylim(-100, 0)
# # ax[3].set_ylim(-100, 0)

# ax[0].grid(linestyle='--')
# ax[1].grid(linestyle='--')
# ax[2].grid(linestyle='--')
# ax[3].grid(linestyle='--')
# ax[4].grid(linestyle='--')
# ax[5].grid(linestyle='--')

# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[3].legend()
# ax[4].legend()
# ax[5].legend()
