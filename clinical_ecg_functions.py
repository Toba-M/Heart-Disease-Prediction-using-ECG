import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
import skimage.io
import skimage.color
import skimage.transform
import skimage.util
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_float

import scipy.ndimage
from scipy.signal import find_peaks, savgol_filter, medfilt, hilbert
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq, ifft
from scipy.stats import entropy
from scipy import signal

from statsmodels.regression.linear_model import yule_walker

import pywt  # PyWavelets for wavelet transforms
from pyhht.emd import EMD
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense, # type: ignore
                                     BatchNormalization, Dropout, 
                                     GlobalAveragePooling1D, InputLayer)

import keras
from keras.models import Sequential
from keras.layers import (Conv1D, BatchNormalization, MaxPooling1D, 
                          GlobalAveragePooling1D, Dense, Dropout, InputLayer)

import antropy as ant
import nolds
import pyhrv.nonlinear as hrv_nl

# Silence warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, InputLayer
def build_ecg_feature_extractor_sequential(input_shape, feature_dim=256):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    
    # First convolution block
    model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    # Second convolution block
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    # Third convolution block
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    # Global pooling to create a fixed-length vector irrespective of input length
    model.add(GlobalAveragePooling1D())
    
    # Feature vector dense layer and dropout for regularization
    model.add(Dense(feature_dim, activation='relu'))
    model.add(Dropout(0.5))
    
    return model

def extract_time_domain_features(ecg_signal):
    ecg_signal = np.asarray(ecg_signal, dtype=np.float64).flatten()
    # Basic statistical features
    mean     = np.mean(ecg_signal)
    median   = np.median(ecg_signal)
    std      = np.std(ecg_signal)
    variance = np.var(ecg_signal)
    iqr      = np.percentile(ecg_signal, 75) - np.percentile(ecg_signal, 25)
    rms      = np.sqrt(np.mean(np.square(ecg_signal)))
    # Signal energy (average squared magnitude)
    energy   = np.sum(np.square(ecg_signal)) / len(ecg_signal)
    # Shannon Entropy based on histogram distribution
    hist, _  = np.histogram(ecg_signal, bins=50, density=True)
    hist    += np.finfo(float).eps  # avoid log(0)
    entropy  = -np.sum(hist * np.log2(hist))
    # Pack into a list (in this exact order)
    values = [
        mean,
        median,
        std,
        variance,
        iqr,
        rms,
        energy,
        entropy
    ]
    return np.array(values)

def detect_peaks(ecg_signal, duration=5, sample_count=500):
    ecg_signal = np.asarray(ecg_signal, dtype=np.float64).squeeze().flatten()
    time = np.linspace(0, duration, sample_count)
    
    # Detect positive and negative peaks.
    positive_peaks, _ = find_peaks(ecg_signal, distance=30, height=np.mean(ecg_signal) + 1)
    negative_peaks, _ = find_peaks(-ecg_signal, distance=30, height=np.mean(-ecg_signal) + 1)
    
    # Compute average amplitudes relative to the mean.
    pos_amplitudes = ecg_signal[positive_peaks] - np.mean(ecg_signal)
    neg_amplitudes = np.mean(ecg_signal) - ecg_signal[negative_peaks]
    
    # Decide on the dominant polarity.
    if len(pos_amplitudes) > 0 and (np.mean(pos_amplitudes) > np.mean(neg_amplitudes)):
        peak_type = "R"   # positive deflections (R) dominate
        peaks = positive_peaks
    else:
        peak_type = "S"   # negative deflections (S) dominate
        peaks = negative_peaks
    
    # Estimate heart rate if at least 2 peaks are detected.
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) * (duration / sample_count)
        heart_rate_bpm = 60 / np.mean(rr_intervals)
    else:
        heart_rate_bpm = None
    
    return {
        "peak_type": peak_type,
        "peak_indices": peaks,
        "heart_rate_bpm": heart_rate_bpm,
        "time": time
    }

def delineate_ecg(ecg_signal, detection_results, duration=5, sample_count=500):
    signal = np.asarray(ecg_signal, dtype=np.float64).squeeze().flatten()
    peak_type = detection_results["peak_type"]
    peak_indices = detection_results["peak_indices"]
    
    delineation = {}
    
    if peak_type == "R":
        # Parameters for the R-based delineation:
        q_points, s_points, p_points, t_points = [], [], [], []
        q_window, s_window = 10, 15           # search windows for Q and S waves around R
        p_offset_start, p_offset_end = 15, 10  # window for P wave search to the left of R
        t_offset_start, t_offset_end = 10, 15  # window for T wave search to the right of R
        
        for r in peak_indices:
            # Q wave: search left of R for minimum (negative deflection).
            q_region = signal[max(0, r - q_window):r]
            if len(q_region) > 0:
                q_index = r - q_window + np.argmin(q_region)
                q_points.append(q_index)
            
            # S wave: search right of R for minimum.
            s_region = signal[r:min(len(signal), r + s_window)]
            if len(s_region) > 0:
                s_index = r + np.argmin(s_region)
                s_points.append(s_index)
            
            # P wave: search to the left of R (offset defined) for maximum (positive deflection).
            p_region = signal[max(0, r - p_offset_start):max(0, r - p_offset_end)]
            if len(p_region) > 0:
                p_index = max(0, r - p_offset_start) + np.argmax(p_region)
                p_points.append(p_index)
            
            # T wave: search to the right of R for maximum.
            t_region = signal[min(len(signal), r + t_offset_start):min(len(signal), r + t_offset_end)]
            if len(t_region) > 0:
                t_index = r + t_offset_start + np.argmax(t_region)
                t_points.append(t_index)
        
        delineation["q_points"] = np.array(q_points)
        delineation["r_points"] = peak_indices  # reference R peaks
        delineation["s_points"] = np.array(s_points)
        delineation["p_points"] = np.array(p_points)
        delineation["t_points"] = np.array(t_points)
    
    elif peak_type == "S":
        # Parameters for the S-based delineation:
        r_points, q_points, p_points, t_points = [], [], [], []
        r_window = 10         # window before S to search for the R wave (positive deflection)
        q_window = 10         # window before R to search for the Q wave (negative deflection)
        p_offset_start = 20   # offsets to locate the P wave (left of S)
        p_offset_end = 15
        t_offset_start = 5    # offsets to locate the T wave (right of S)
        t_offset_end = 11
        
        for s_idx in peak_indices:
            # R wave: search a window before S for maximum.
            r_start = max(0, s_idx - r_window)
            r_region = signal[r_start:s_idx]
            if len(r_region) > 0:
                local_r = np.argmax(r_region)
                r_idx = r_start + local_r
                r_points.append(r_idx)
            else:
                continue  # Skip if the region is empty.
            
            # Q wave: search before the detected R for minimum.
            q_start = max(0, r_idx - q_window)
            q_region = signal[q_start:r_idx]
            if len(q_region) > 0:
                local_q = np.argmin(q_region)
                q_idx = q_start + local_q
                q_points.append(q_idx)
            
            # P wave: search left of S using offsets.
            p_start = max(0, s_idx - p_offset_start)
            p_end = max(0, s_idx - p_offset_end)
            p_region = signal[p_start:p_end]
            if len(p_region) > 0:
                p_idx = p_start + np.argmax(p_region)
                p_points.append(p_idx)
            
            # T wave: search right of S using offsets.
            t_start = s_idx + t_offset_start
            t_end = min(len(signal), s_idx + t_offset_end)
            t_region = signal[t_start:t_end]
            if len(t_region) > 0:
                t_idx = t_start + np.argmax(t_region)
                t_points.append(t_idx)
        
        delineation["q_points"] = np.array(q_points)
        delineation["r_points"] = np.array(r_points)  # detected R preceding S
        delineation["s_points"] = peak_indices         # reference S peaks
        delineation["p_points"] = np.array(p_points)
        delineation["t_points"] = np.array(t_points)
    
    else:
        raise ValueError("Unknown peak type detected.")
        
    return delineation

def compute_ecg_features(signal, time, delineation):
    features = {}

    # Extract points
    r_pts = delineation["r_points"]
    q_pts = delineation["q_points"]
    s_pts = delineation["s_points"]
    p_pts = delineation["p_points"]
    t_pts = delineation["t_points"]

    # Time conversions
    t_r = time[r_pts]
    t_q = time[q_pts]
    t_s = time[s_pts]
    t_p = time[p_pts]
    t_t = time[t_pts]

    # --- Time Intervals ---
    rr_intervals = np.diff(t_r)
    features["mean_rr"] = np.mean(rr_intervals)
    features["heart_rate"] = 60 / features["mean_rr"] if features["mean_rr"] else None

    pr_intervals = t_r[:len(t_p)] - t_p
    features["mean_pr"] = np.mean(np.abs(pr_intervals))

    qt_intervals = t_t[:len(t_q)] - t_q[:len(t_t)]
    features["mean_qt"] = np.mean(np.abs(qt_intervals))

    qrs_durations = t_s[:len(t_q)] - t_q[:len(t_s)]
    features["mean_qrs_duration"] = np.mean(np.abs(qrs_durations))

    # --- P wave durations ---
    p_durations = []
    for p in p_pts:
        start_idx = max(0, p - 30)
        end_idx = min(len(signal), p + 30)
        region = signal[start_idx:end_idx]
        time_region = time[start_idx:end_idx]

        baseline = np.median(region[:5])
        peak_amp = signal[p]
        polarity = np.sign(peak_amp)

        if polarity >= 0:
            half_max = (abs(peak_amp - baseline)) / 2 + baseline
            crossings = np.where(region < half_max)[0]
        else:
            half_max = baseline - (abs(peak_amp - baseline)) / 2
            crossings = np.where(region > half_max)[0]

        before_peak = crossings[crossings < (p - start_idx)]
        after_peak = crossings[crossings > (p - start_idx)]

        if len(before_peak) > 0 and len(after_peak) > 0:
            start = before_peak[-1]
            end = after_peak[0]
            p_durations.append(time_region[end] - time_region[start])

    features["mean_p_duration"] = np.mean(p_durations) if p_durations else None

    # --- T wave durations ---
    t_durations = []
    for t in t_pts:
        start_idx = max(0, t - 30)
        end_idx = min(len(signal), t + 30)
        region = signal[start_idx:end_idx]
        time_region = time[start_idx:end_idx]

        baseline = np.median(region[:5])
        peak_amp = signal[t]
        polarity = np.sign(peak_amp)

        if polarity >= 0:
            half_max = (abs(peak_amp - baseline)) / 2 + baseline
            crossings = np.where(region < half_max)[0]
        else:
            half_max = baseline - (abs(peak_amp - baseline)) / 2
            crossings = np.where(region > half_max)[0]

        before_peak = crossings[crossings < (t - start_idx)]
        after_peak = crossings[crossings > (t - start_idx)]

        if len(before_peak) > 0 and len(after_peak) > 0:
            start = before_peak[-1]
            end = after_peak[0]
            t_durations.append(time_region[end] - time_region[start])

    features["mean_t_duration"] = np.mean(t_durations) if t_durations else None

    # --- Morphological Features ---
    features["mean_amplitudes"] = {
        "P": np.mean(signal[p_pts]) if len(p_pts) else None,
        "Q": np.mean(signal[q_pts]) if len(q_pts) else None,
        "R": np.mean(signal[r_pts]) if len(r_pts) else None,
        "S": np.mean(signal[s_pts]) if len(s_pts) else None,
        "T": np.mean(signal[t_pts]) if len(t_pts) else None
    }

    # Slopes
    if len(q_pts) and len(r_pts):
        slopes_qr = (signal[r_pts[:len(q_pts)]] - signal[q_pts]) / (t_r[:len(q_pts)] - t_q)
        features["mean_qr_slope"] = np.mean(slopes_qr)
    else:
        features["mean_qr_slope"] = None

    if len(r_pts) and len(s_pts):
        slopes_rs = (signal[s_pts[:len(r_pts)]] - signal[r_pts[:len(s_pts)]]) / (t_s[:len(r_pts)] - t_r[:len(s_pts)])
        features["mean_rs_slope"] = np.mean(slopes_rs)
    else:
        features["mean_rs_slope"] = None

    # Area under segments
    def area_under(start_idx, end_idx):
        return np.trapz(signal[start_idx:end_idx], time[start_idx:end_idx]) if end_idx > start_idx else 0

    areas_qrs = [area_under(q, s) for q, s in zip(q_pts, s_pts) if s > q]
    areas_p = [area_under(p-2, p+3) for p in p_pts if p-2 >= 0 and p+3 < len(signal)]
    areas_t = [area_under(t-3, t+4) for t in t_pts if t-3 >= 0 and t+4 < len(signal)]

    features["mean_area_qrs"] = np.mean(areas_qrs) if areas_qrs else None
    features["mean_area_p"] = np.mean(areas_p) if areas_p else None
    features["mean_area_t"] = np.mean(areas_t) if areas_t else None

    return features

def extract_morphological_features(sig, duration=5, sample_count=500):
    import numpy as np

    time = np.linspace(0, duration, sample_count)

    det = detect_peaks(sig, duration=duration, sample_count=sample_count)
    deline = delineate_ecg(sig, det, duration=duration, sample_count=sample_count)
    features = compute_ecg_features(sig, time, deline)

    def stats(indices):
        return [
            len(indices),                         # count
            np.mean(indices) if len(indices) > 0 else 0,
            np.std(indices) if len(indices) > 0 else 0
        ]

    values = [
        det['heart_rate_bpm'] if det['heart_rate_bpm'] is not None else 0,

        *stats(deline['q_points']),
        *stats(deline['r_points']),
        *stats(deline['s_points']),
        *stats(deline['p_points']),
        *stats(deline['t_points']),

        features.get("mean_rr", 0),
        features.get("heart_rate", 0),
        features.get("mean_pr", 0),
        features.get("mean_qt", 0),
        features.get("mean_qrs_duration", 0),
        features.get("mean_p_duration", 0),
        features.get("mean_t_duration", 0),

        features["mean_amplitudes"].get("P", 0),
        features["mean_amplitudes"].get("Q", 0),
        features["mean_amplitudes"].get("R", 0),
        features["mean_amplitudes"].get("S", 0),
        features["mean_amplitudes"].get("T", 0),

        features.get("mean_qr_slope", 0),
        features.get("mean_rs_slope", 0),
        features.get("mean_area_qrs", 0),
        features.get("mean_area_p", 0),
        features.get("mean_area_t", 0)
    ]

    return np.array(values)

def compute_wavelet_features(sig, fs):
    """
    Compute wavelet-based features.
    - DWT: energies of approximation and detail coefficients (levels).
    - CWT: dominant frequency from the scalogram and its energy.
    """
    # Discrete Wavelet Transform (multi-level decomposition)
    wavelet = 'db4'
    level = 5
    sig = np.asarray(sig, dtype=np.float64).squeeze().flatten()
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    # Energy of coefficients at each level (approximation + details)
    band_energies = [np.sum(c**2) for c in coeffs]
    # Continuous Wavelet Transform (using Mexican Hat wavelet)
    widths = np.arange(1, 91)
    cwt_matrix, freqs = pywt.cwt(sig, widths, 'mexh', sampling_period=1.0/fs)
    # Compute energy at each scale
    scale_energy = np.sum(np.abs(cwt_matrix)**2, axis=1)
    max_scale_idx = np.argmax(scale_energy)
    dominant_freq = freqs[max_scale_idx] if freqs is not None else 0.0
    max_energy = scale_energy[max_scale_idx]
    # Concatenate DWT energies with dominant CWT frequency and energy
    return np.concatenate([band_energies, np.array([dominant_freq, max_energy])])

def compute_stft_features(sig, fs):
    """
    Compute STFT-based features from the spectrogram.
    Returns [dom_freq, dom_power, total_spectrogram_power].
    """
    sig = np.asarray(sig, dtype=np.float64).squeeze().flatten()
    f, t, Z = signal.stft(sig, fs)
    spectrogram = np.abs(Z)**2  # power spectrogram
    avg_spectrum = np.mean(spectrogram, axis=1)  # mean power at each frequency over time
    total_power = np.sum(avg_spectrum)
    # Dominant frequency from average spectrum
    idx = np.argmax(avg_spectrum[1:]) + 1 if len(avg_spectrum) > 1 else 0
    dom_freq = f[idx]
    dom_power = avg_spectrum[idx]
    return np.array([dom_freq, dom_power, total_power])

def compute_hht_features(sig, fs):
    # Ensure input is a NumPy array for processing
    sig = np.asarray(sig, dtype=np.float64).squeeze().flatten()
    
    # Perform Empirical Mode Decomposition to get up to 3 IMFs
    decomposer = EMD(sig, n_imfs=3)
    imfs = decomposer.decompose()  # imfs is an array of shape (num_imfs, signal_length)
    num_imfs = imfs.shape[0] if imfs.size > 0 else 0  # Number of IMFs extracted (could be 0 if none)
    
    features = [num_imfs]  # Start feature list with the number of IMFs
    # Loop over each IMF (up to 3) and compute features
    for i in range(min(3, num_imfs)):
        imf = imfs[i]  # i-th IMF (array of length N)
        # Compute the analytic signal using Hilbert transform
        analytic_signal = hilbert(imf)
        amp_env = np.abs(analytic_signal)               # Amplitude envelope of the IMF
        inst_phase = np.unwrap(np.angle(analytic_signal))  # Unwrapped instantaneous phase
        
        # Compute instantaneous frequency (in Hz) from phase derivative
        # Note: np.diff reduces length by 1, so inst_freq has length N-1
        if len(inst_phase) < 2:
            # If IMF is too short (edge case), define frequency as 0
            mean_inst_freq = 0.0
        else:
            inst_freq = np.diff(inst_phase) * (fs / (2.0 * np.pi))  # Instantaneous frequency array (length N-1)
            # Use amplitude at the corresponding points as weights (amp_env[1:] has length N-1 to match inst_freq)
            weights = amp_env[1:]
            total_weight = np.sum(weights)
            if total_weight == 0:
                # Avoid division by zero if the IMF has zero amplitude
                mean_inst_freq = 0.0
            else:
                # Weighted mean instantaneous frequency (amplitude-weighted)
                mean_inst_freq = np.sum(inst_freq * weights) / total_weight
        
        # Compute energy of the IMF (sum of squared signal values)
        energy = np.sum(imf ** 2)
        
        # Append the features for this IMF
        features.append(mean_inst_freq)
        features.append(energy)
    
    return np.array(features)

def compute_ar_features(sig, order=10):
    """
    Fit an AR model and return coefficients and error variance.
    Returns [ar_coeff1, ar_coeff2, ..., ar_coeffN, error_variance].
    """
    sig = np.asarray(sig, dtype=np.float64).squeeze().flatten()
    sig = sig - np.mean(sig)
    # Solve Yule-Walker equations for AR coefficients
    coeffs, noise_var = yule_walker(sig, order=order, method="mle")
    return np.concatenate([coeffs, np.array([noise_var])])

def compute_cepstral_features(sig, fs):
    """
    Compute real cepstral coefficients of the signal.
    Returns an array of the first 10 cepstral coefficients (or fewer if signal is short).
    """
    sig = np.asarray(sig, dtype=np.float64).squeeze().flatten()
    sig = sig - np.mean(sig)
    N = len(sig)
    spectrum = fft(sig)
    mag = np.abs(spectrum) + 1e-10
    log_mag = np.log(mag)
    cepstrum = np.real(ifft(log_mag))
    cepstrum = cepstrum[:N//2]            # take first half of cepstrum (real cepstrum is symmetric)
    num_coeff = min(10, len(cepstrum))
    return cepstrum[:num_coeff]
def compute_spectral_entropy(sig, fs):
    """
    Compute spectral entropy of the signal.
    Returns [spectral_entropy], with entropy in bits.
    """
    sig = np.asarray(sig, dtype=np.float64).squeeze().flatten()
    f, Pxx = signal.periodogram(sig, fs)
    Pxx = Pxx + 1e-12  # avoid zero probabilities
    Pxx_norm = Pxx / np.sum(Pxx)
    # Shannon entropy of the normalized PSD
    spec_entropy = -np.sum(Pxx_norm * np.log2(Pxx_norm))
    return np.array([spec_entropy])

def compute_fft_features(sig, fs):
    """
    Compute frequency-domain features using the Fast Fourier Transform (FFT).
    Returns [peak_freq, peak_magnitude, spectral_centroid, total_power].
    """
    sig = np.asarray(sig, dtype=np.float64).squeeze().flatten()
    sig = sig - np.mean(sig)  # remove DC offset
    N = len(sig)
    X = fft(sig)
    freqs = fftfreq(N, d=1/fs)
    # Consider only the non-negative frequencies (real signal FFT is symmetric)
    half_N = N//2 + 1
    X = X[:half_N]
    freqs = freqs[:half_N]
    magnitudes = np.abs(X)
    # Total spectral power (sum of squared magnitudes)
    power = np.sum(magnitudes**2)
    # Find dominant frequency (exclude DC component at index 0)
    if half_N > 1:
        peak_idx = np.argmax(magnitudes[1:]) + 1
    else:
        peak_idx = 0
    peak_freq = freqs[peak_idx]
    peak_mag = magnitudes[peak_idx]
    # Spectral centroid (power-weighted average frequency)
    centroid = 0.0
    if power > 1e-8:
        centroid = np.sum(freqs * (magnitudes**2)) / power
    return np.array([peak_freq, peak_mag, centroid, power])

def compute_psd_features(sig, fs):
    """
    Compute features from Power Spectral Density (PSD).
    Returns [peak_freq_welch, peak_psd_welch, total_power_welch,
             peak_freq_periodogram, peak_psd_periodogram, total_power_periodogram].
    """
    # Welch's PSD (averaged periodograms)
    sig = np.asarray(sig, dtype=np.float64).squeeze().flatten()
    f_welch, Pxx_welch = signal.welch(sig, fs)
    # Single periodogram PSD
    f_per, Pxx_per = signal.periodogram(sig, fs)
    # Total powers (area under PSD curve)
    power_welch = np.sum(Pxx_welch)
    power_per   = np.sum(Pxx_per)
    # Dominant frequency from Welch PSD
    idx_welch = np.argmax(Pxx_welch[1:]) + 1 if len(Pxx_welch) > 1 else 0
    peak_freq_welch = f_welch[idx_welch]
    peak_psd_welch  = Pxx_welch[idx_welch]
    # Dominant frequency from single periodogram
    idx_per = np.argmax(Pxx_per[1:]) + 1 if len(Pxx_per) > 1 else 0
    peak_freq_per = f_per[idx_per]
    peak_psd_per  = Pxx_per[idx_per]
    return np.array([peak_freq_welch, peak_psd_welch, power_welch,
                     peak_freq_per, peak_psd_per, power_per])
def extract_frequency_features(sig, fs, ar_order=10):
    """
    Extract frequency-domain features from a single-lead ECG signal.
    Combines features from FFT, PSD, Wavelet, STFT, HHT, AR, Cepstral, and Spectral Entropy into one feature vector.
    """
    sig = np.asarray(sig, dtype=np.float64).squeeze().flatten()
    feature_list = [
        compute_fft_features(sig, fs),
        compute_psd_features(sig, fs),
        compute_wavelet_features(sig, fs),
        compute_stft_features(sig, fs),
        compute_hht_features(sig, fs),
        compute_ar_features(sig, order=ar_order),
        compute_cepstral_features(sig, fs),
        compute_spectral_entropy(sig, fs)
    ]
    return np.concatenate(feature_list)

def extract_features_batch(signals, fs, ar_order=10):
    
    return np.vstack([extract_frequency_features(sig, fs, ar_order=ar_order) for sig in signals])

def enhance_contrast(img):
    # Validate input
    if img is None or img.size == 0:
        raise ValueError("Invalid input image")
    # Create CLAHE object with appropriate parameters
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
    # Apply CLAHE to enhance contrast
    enhanced_img = clahe.apply(img)
    return enhanced_img

def normalize_image(img):
    if img is None or img.size == 0:
        raise ValueError("Invalid input image")   
    # Handle potential division by zero and negative values
    img_float = img.astype(np.float32)
    img_min = np.min(img_float)
    img_max = np.max(img_float)
    # Prevent division by zero
    if img_min == img_max:
        return np.zeros_like(img, dtype=np.uint8)
    # Normalize to 0-255 range
    img_normalized = (img_float - img_min) / (img_max - img_min) * 255
    return np.uint8(img_normalized)

def extract_valid_ecg_contours(edges, min_area=0.05, min_width_fraction=0.005, debug=False):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if debug:
            print("❌ No contours found in the image.")
        return []
    # Filter contours by area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    #print(f"👍{len(filtered_contours)} met the criteria")
    if not filtered_contours:
        if debug:
            print("❌ No contours found exceeding the minimum area threshold.")
        return []
    image_width = edges.shape[1]
    candidate_contours = [cnt for cnt in filtered_contours if cv2.boundingRect(cnt)[2] > image_width * min_width_fraction]
    if debug:
        #print(f"✅ {len(candidate_contours)} contours met the width fraction criteria.")
    # Return all valid contours (empty list if none)
        return candidate_contours if candidate_contours else filtered_contours


def map_contours_to_signal(contours, img_shape, num_points=500, interp_kind='linear'):
    all_points = []
    for contour in contours:
        points = np.squeeze(contour)  # Remove extra dimensions
        if len(points.shape) == 2 and points.shape[1] == 2:
            all_points.append(points)
    if not all_points:
        raise ValueError("No valid points found in contours.")
    # Merge all points into a single array
    all_points = np.vstack(all_points)
    # Sort points by x-coordinate
    all_points = all_points[np.argsort(all_points[:, 0])]
    x_vals = all_points[:, 0]
    y_vals = img_shape[0] - all_points[:, 1]  # Invert y-axis for correct ECG alignment
    # Remove duplicate x-values by averaging corresponding y-values
    unique_x = np.unique(x_vals)
    averaged_y = np.array([np.mean(y_vals[x_vals == x]) for x in unique_x])
    # Interpolate signal
    f_interp = interp1d(unique_x, averaged_y, kind=interp_kind, fill_value='extrapolate')
    x_new = np.linspace(unique_x.min(), unique_x.max(), num=num_points)
    y_new = f_interp(x_new)
    # Apply a Savitzky-Golay filter for smoothing
    window_length = min(9, num_points if num_points >= 9 else (num_points // 2) * 2 + 1)
    y_smooth = savgol_filter(y_new, window_length=window_length, polyorder=2)
    return x_new, y_smooth

def baseline_correction_ecg(signal, kernel_size=81):
    baseline = medfilt(signal, kernel_size=kernel_size)
    return signal - baseline

def extract_nonlinear_features(ecg_signal, fs=500, detect_peaks_fn=None):
    # Ensure 1D float64 array
    plt.ioff()
    sig = np.asarray(ecg_signal, dtype=np.float64).squeeze().flatten()
    duration = len(sig) / fs
    features = {}

    # --- Entropy Features ---
    entropy_funcs = [
        ('ApEn',       ant.app_entropy,     {'order':2}),
        ('SampEn',     ant.sample_entropy,  {'order':2}),
        ('PermEn',     ant.perm_entropy,    {'order':3, 'normalize':True}),
        ('SpectralEn', ant.spectral_entropy,{'sf':fs, 'method':'welch', 'normalize':True})
    ]
    for name, fn, kwargs in entropy_funcs:
        try:
            features[name] = float(fn(sig, **kwargs))
        except Exception:
            features[name] = np.nan

    # --- Fractal / Complexity Features ---
    fractal_funcs = [
        ('PetrosianFD', ant.petrosian_fd,   {}),
        ('KatzFD',      ant.katz_fd,        {}),
        ('HiguchiFD',   ant.higuchi_fd,     {'kmax':10}),
        ('CorrDim',     nolds.corr_dim,     {'emb_dim':5}),
        ('HurstExp',    nolds.hurst_rs,     {}),
        ('DFA_alpha',   nolds.dfa,          {}),
        ('LyapunovExp', lambda x: nolds.lyap_r(x, emb_dim=10, tau=1), {})
    ]
    for name, fn, kwargs in fractal_funcs:
        try:
            features[name] = float(fn(sig, **kwargs))
        except Exception:
            features[name] = np.nan

    # --- R-peak Detection ---
    if detect_peaks_fn:
        det = detect_peaks_fn(sig, duration=duration, sample_count=len(sig))
        peaks_idx = det.get("peak_indices", np.array([], dtype=int))
    else:
        peaks_idx, _ = find_peaks(sig, distance=0.2 * fs)

    rpeaks_sec = peaks_idx.astype(np.float64) / fs

    # --- Poincaré Plot Features (no plotting) ---
    try:
        poi = hrv_nl.poincare(nni=None, rpeaks=rpeaks_sec, show=False)
        plt.close('all')  # close any hidden figures
        features['Poincare_SD1']           = float(poi['sd1'])
        features['Poincare_SD2']           = float(poi['sd2'])
        features['Poincare_SD2_SD1_ratio'] = float(poi['sd_ratio'])
        features['Poincare_ellipse_area']  = float(poi['ellipse_area'])
    except Exception:
        features['Poincare_SD1'] = np.nan
        features['Poincare_SD2'] = np.nan
        features['Poincare_SD2_SD1_ratio'] = np.nan
        features['Poincare_ellipse_area']  = np.nan
    features = list(features.values())
    arr = np.array(features)
    arr[~np.isfinite(arr)] = np.nan
    return arr