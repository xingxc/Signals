from scipy.fft import fft, ifft
import numpy as np
import math


class windows():
    def __init__(self, N, beta=14):
        self.N = N
        self.beta = beta
        self.window_dict = {'rect': np.ones((self.N)),
                            'hanning': np.hanning(self.N),
                            'hamming': np.hamming(self.N),
                            'blackman': np.blackman(self.N),
                            'kaiser': np.kaiser(self.N, self.beta)}


class fourier_transform():

    def __init__(self, signal_X, signal_Y):
        # signal data
        self.signal_X = signal_X
        self.signal_Y = signal_Y
        self.Fs = 1/np.diff(np.array(signal_X)).mean()
        self.N = self.signal_Y.__len__()
        # fft data
        self.signal_F = None
        self.signal_bins = None
        self.signal_freq = None
        self.signal_phase = None
        self.signal_mag = None
        self.signal_dB = None

        self.signal_F_thr = None
        self.signal_phase_thr = None
        self.signal_mag_thr = None
        self.signal_dB_thr = None

    def signal_threshold(self, cut_off, write='False'):
        threshold = max(abs(self.signal_F)/cut_off)
        signal_mask = abs(self.signal_F) < threshold
        filtered = self.signal_F
        filtered[signal_mask] = 0
        if write:
            self.signal_F_thr = filtered
            self.signal_mag_thr, self.signal_phase_thr = self.imag_to_phase(
                self.signal_F_thr)
            self.signal_dB_thr = self.to_dB(self.signal_mag_thr)
        return filtered

    def to_dB(self, mag=None):
        if mag is None:
            mag = self.signal_mag

        mag = 20*np.log10(mag)

        return mag - mag.max()

    def nxt_pow_2_shift(self):
        return (lambda x: 1 << (x-1).bit_length())(self.signal_Y.__len__())

    def imag_to_phase(self, X):
        mag = np.absolute(X)
        angle = np.arctan2(X.imag, X.real)
        # angle = np.angle(X)
        return mag, angle

    def DFT_analyzing_function(self, k, m, N):
        '''
        input of k, 
        '''
        return np.exp(-1j * (2*math.pi*k / N) * m)

    def DFT(self, signal=None, N=None):
        '''
        Basic DFT algorit
        '''
        if N is None:
            N = self.N
        if signal is None:
            signal = self.signal_Y

        Y_values = []

        for k in range(N):
            # k_values.append(k)
            y_temp = []
            for m in range(N):

                c = self.DFT_analyzing_function(k, m, N)
                y_temp.append(self.signal_Y[m]*c)

            Y_values.append(np.array(y_temp).sum())

        # k_values = np.array(k_values)
        Y_values = np.array(Y_values)

        return Y_values

    def FFT_FFT(self, signal_in=None,  N=None, solver='scipy', norm=None):
        '''
        Input
        signal_in : input signal
        N : length of the fft solver, will pad with zero if N greater than length of signal
        solver: specify which fft solver {numpy, scipy}
        norm: specify the kind of normalization on fft {ortho}
        '''
        if N is None:
            N = self.N

        if N == 'pow2':
            N = self.nxt_pow_2_shift()

        if signal_in is None:
            signal_in = self.signal_Y

        FFT_types = {
            'scipy': lambda signal, num: np.fft.fftshift(fft(signal, num, norm=norm)),
            'numpy': lambda signal, num: np.fft.fftshift(np.fft.fft(signal, num, norm=norm)),
        }

        self.signal_F = FFT_types[solver](signal_in, N)
        self.signal_bins = np.fft.fftshift(np.fft.fftfreq(N, N/self.Fs))
        self.signal_freq = np.fft.fftshift(np.fft.fftfreq(N, 1/self.Fs))

        # signal_freq = np.arange(0, int(N/2)) * self.Fs/(N)
        # signal_F = signal_F[0:int(N/2)]

        self.signal_mag, self.signal_phase = self.imag_to_phase(self.signal_F)
        self.signal_dB = self.to_dB()
        return 0

    def signal_return(self):
        '''
        returns all data stored
        '''
        result = {'frequency': self.signal_freq,
                  'signal': self.signal_F,
                  'mag': self.signal_mag,
                  'mag_dB': self.signal_dB,
                  'phase': self.signal_phase}

        return result


def create_signal(Fs, time_start, time_duration, amplitude, signal_freq, signal_phase, signal_type='sin'):
    '''
    return : sin or cos signal

    Fs : sampling frequency
    time_start : starting time of the signal
    time_duration: total duration of the signal
    amplitude : amplitude of the signal
    signal_freq : frequency of the signal
    signal_phase : phase of the signal
    signal_type : sine or cosine
    '''
    amplitude = np.array(amplitude)
    signal_freq = np.array(signal_freq)
    signal_phase = np.array(signal_phase)

    signal_types = {'sin': lambda x: np.sin(x),
                    'cos': lambda x: np.cos(x)}

    if amplitude.__len__() == signal_freq.__len__() == signal_phase.__len__():

        signal_X = np.arange(time_start, time_duration, 1/Fs)

        signals = []

        for i in range(amplitude.__len__()):
            signals.append(amplitude[i] * signal_types[signal_type]
                           (2*math.pi * signal_freq[i]*signal_X + signal_phase[i]))

        signal_Y = 0
        for item in signals:
            signal_Y += item

        return signal_X, signal_Y

    else:
        return 'must have equivalent number of amplitude, frequency and phase/'


def rect_gen(array_in, min_value=None, max_value=None):
    '''
    return : rectangular signal
    array_in : x axis of the signal
    min_value : minimum value within the array
    max_value : maximum value within the array
    '''

    if min_value is None:
        min_value = array_in.min()

    if max_value is None:
        max_value = array_in.max()

    array_min = array_in > min_value
    array_max = array_in < max_value
    array_out = np.array([])
    for i in range(array_min.__len__()):
        array_out = np.append(array_out, array_min[i] and array_max[i])

    return array_out.astype(int)
