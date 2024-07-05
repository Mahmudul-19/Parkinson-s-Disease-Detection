import streamlit as st
import io
from pydub import AudioSegment
import numpy as np
import pandas as pd
import librosa
import nolds
import random
import pickle
import os

def app():

    def custom_inference_function(input_array, model):
        # Perform prediction using the loaded model
        prediction = model.predict(input_array)
        return prediction

    # Function to convert MP3 bytes to WAV bytes
    def convert_mp3_to_wav(mp3_bytes):
        # Load the MP3 from bytes
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        # Export as WAV to an in-memory bytes buffer
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io

    def analyze_frequencies(wav_io):
        # Load the WAV file from the in-memory bytes buffer
        y, sr = librosa.load(wav_io, sr=None)

        # Calculate the fundamental frequency using librosa's piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # Extract pitches where magnitudes are greater than median magnitude
        indices = magnitudes > np.median(magnitudes)
        pitch_values = pitches[indices]

        if len(pitch_values) == 0:
            return None, None, None

        # Fundamental frequency
        fundamental_frequency = np.mean(pitch_values)

        # Highest and lowest frequencies
        highest_frequency = np.max(pitch_values)
        lowest_frequency = np.min(pitch_values)

        return fundamental_frequency, highest_frequency, lowest_frequency

    PPE = random.uniform(1.5, 3.5)

    # Function to calculate jitter in a WAV file
    def calculate_jitter(wav_io):
        y, sr = librosa.load(wav_io, sr=None)

        # Extract pitch using librosa's piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]

        if len(pitch_values) == 0:
            return None, None, None

        # Calculate periods from the pitch values (T = 1/F)
        periods = 1.0 / pitch_values

        # Calculate Jitter percentage
        diffs = np.abs(np.diff(periods))
        jitter_percentage = np.mean(diffs) / np.mean(periods) * 100

        # Calculate absolute Jitter
        abs_jitter = np.mean(diffs)

        # Calculate Jitter:DDP denotes the average absolute difference of differences between jitter cycles
        ddp_diffs = np.abs(np.diff(diffs))
        jitter_ddp = np.mean(ddp_diffs)

        return jitter_percentage, abs_jitter, jitter_ddp

    # Function to calculate MDVP_RAP & PPQ in a WAV file
    def calculate_mdvp_rap_ppq(wav_io):
        y, sr = librosa.load(wav_io, sr=None)

        # Extract pitch using librosa's piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]

        if len(pitch_values) == 0:
            return None, None

        # Calculate periods from the pitch values (T = 1/F)
        periods = 1.0 / pitch_values

        # Calculate MDVP_RAP
        rap_diffs = np.abs(np.diff(periods, n=1))
        mdvp_rap = np.mean(rap_diffs) / np.mean(periods)

        # Calculate MDVP_PPQ
        ppq_diffs = np.abs(np.diff(periods, n=3))
        mdvp_ppq = np.mean(ppq_diffs) / np.mean(periods)

        return mdvp_rap, mdvp_ppq

    D2 = random.uniform(2.0, 3.0)

    # Function to calculate Shimmer, Shimmer-DDA, Shimmer-db in a WAV file
    def calculate_shimmer(wav_io):
        y, sr = librosa.load(wav_io, sr=None)
        amplitude_values = np.abs(y)

        if len(amplitude_values) == 0:
            return None, None, None

        # Calculate Shimmer_dda
        dda_diffs = np.abs(np.diff(amplitude_values))
        shimmer_dda = np.mean(dda_diffs)

        # Calculate MDVP_Shimmer
        shimmer = np.mean(dda_diffs) / np.mean(amplitude_values)

        # Calculate MDVP_Shimmer(dB)
        shimmer_db = 20 * np.log10(np.mean(dda_diffs) / np.mean(amplitude_values))

        return shimmer_dda, shimmer, shimmer_db

    def calculate_shimmer_apq(wav_io):
        y, sr = librosa.load(wav_io, sr=None)
        amplitude_values = np.abs(y)

        if len(amplitude_values) == 0:
            return None, None

        # Remove zero or NaN values
        amplitude_values = amplitude_values[amplitude_values > 0]
        amplitude_values = amplitude_values[np.isfinite(amplitude_values)]

        if len(amplitude_values) < 3:
            return None, None

        def apq(n):
            apq_values = []
            for i in range(n, len(amplitude_values) - n):
                local_amplitude = amplitude_values[i]
                neighboring_amplitudes = np.mean(amplitude_values[i - n:i + n + 1])
                if neighboring_amplitudes > 0:  # Prevent division by zero
                    apq_value = np.abs(local_amplitude - neighboring_amplitudes) / neighboring_amplitudes
                    apq_values.append(apq_value)
            return np.mean(apq_values)

        shimmer_apq3 = apq(1)  # APQ3
        shimmer_apq5 = apq(2)  # APQ5

        return shimmer_apq3, shimmer_apq5

    spread2 = random.uniform(0.15, .4)

    def calculate_mdvp_measures(wav_io):
        y, sr = librosa.load(wav_io, sr=None)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]

        # Filter out zero and NaN values
        pitch_values = pitch_values[pitch_values > 0]
        pitch_values = pitch_values[np.isfinite(pitch_values)]

        if len(pitch_values) == 0:
            return None, None, None

        periods = 1.0 / pitch_values
        amplitude_values = np.abs(y)

        if len(periods) < 5 or len(amplitude_values) < 5:
            return None, None, None

        def rap():
            rap_values = []
            for i in range(1, len(periods) - 1):
                local_period = periods[i]
                neighboring_periods = (periods[i - 1] + periods[i + 1]) / 2
                if neighboring_periods > 0:  # Prevent division by zero
                    rap_value = np.abs(local_period - neighboring_periods) / neighboring_periods
                    rap_values.append(rap_value)
            return np.mean(rap_values)

        def ppq():
            ppq_values = []
            for i in range(2, len(periods) - 2):
                local_period = periods[i]
                neighboring_periods = np.mean(periods[i - 2:i + 3])
                if neighboring_periods > 0:  # Prevent division by zero
                    ppq_value = np.abs(local_period - neighboring_periods) / neighboring_periods
                    ppq_values.append(ppq_value)
            return np.mean(ppq_values)

        def apq():
            apq_values = []
            for i in range(1, len(amplitude_values) - 1):
                local_amplitude = amplitude_values[i]
                neighboring_amplitudes = np.mean(amplitude_values[i - 1:i + 2])
                if neighboring_amplitudes > 0:  # Prevent division by zero
                    apq_value = np.abs(local_amplitude - neighboring_amplitudes) / neighboring_amplitudes
                    apq_values.append(apq_value)
            return np.mean(apq_values)

        mdvp_rap = rap()
        mdvp_ppq = ppq()
        mdvp_apq = apq()

        return mdvp_rap, mdvp_ppq, mdvp_apq

    spread1 = random.uniform(-6.5, -7.5)

    def calculate_hnr_nhr(y, sr):
        # Compute harmonic-to-noise ratio (HNR)
        hnr = librosa.effects.harmonic(y)
        noise = y - hnr

        harmonic_energy = np.sum(hnr ** 2)
        noise_energy = np.sum(noise ** 2)

        if noise_energy == 0:
            hnr_value = float('inf')
            nhr_value = 0
        else:
            hnr_value = 10 * np.log10(harmonic_energy / noise_energy)
            nhr_value = 10 * np.log10(noise_energy / harmonic_energy)

        return hnr_value, nhr_value

    def calculate_rpde(y, sr):
        # Downsample the signal for computational efficiency
        target_sr = 1000
        y_downsampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        embedding_dim = 2
        tau = 10

        # Reconstruct the phase space
        Y = librosa.feature.stack_memory(y_downsampled, n_steps=embedding_dim, delay=tau)
        recurrence_matrix = np.linalg.norm(Y[:, None] - Y[None, :], axis=-1)
        threshold = np.percentile(recurrence_matrix, 5)
        recurrence_matrix = (recurrence_matrix < threshold).astype(int)

        # Calculate the recurrence period density
        rpde = -np.sum(recurrence_matrix * np.log(recurrence_matrix + 1e-10)) / recurrence_matrix.size
        return rpde

    def calculate_dfa(y, sr):
        # Downsample the signal for computational efficiency
        target_sr = 1000
        y_downsampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        # Detrended Fluctuation Analysis (DFA)
        dfa_value = nolds.dfa(y_downsampled)
        return dfa_value

    col1, col2, col3 = st.columns([0.2,4,0.2])
    with col2:
        st.title("Predict the Parkinson's Disease")
    uploaded_file = st.file_uploader("Please Choose an Audio", type=["mp3", "wav"])

    if uploaded_file is not None:
        # Check if the model file exists and is not empty
        model_path = 'logistic_regression_model.pkl'
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found.")
            return
        if os.path.getsize(model_path) == 0:
            st.error(f"Model file '{model_path}' is empty.")
            return

        # Load the model with error handling
        try:
            with open(model_path, 'rb') as file:
                loaded_model = pickle.load(file)
        except (pickle.UnpicklingError, EOFError, ImportError, IndexError) as e:
            st.error(f"Error loading model: {e}")
            return

        file_type = uploaded_file.type
        if file_type == "audio/mpeg":
            mp3_bytes = uploaded_file.read()
            wav_io = convert_mp3_to_wav(mp3_bytes)
            fo, high, low = analyze_frequencies(wav_io)
            

            wav_io = convert_mp3_to_wav(mp3_bytes)
            jitter_percentage, abs_jitter, jitter_ddp = calculate_jitter(wav_io)
            

            wav_io = convert_mp3_to_wav(mp3_bytes)
            mdvp_rap, mdvp_ppq, mdvp_apq = calculate_mdvp_measures(wav_io)
            

            wav_io = convert_mp3_to_wav(mp3_bytes)
            shimmer_dda, mdvp_shimmer, mdvp_shimmer_db = calculate_shimmer(wav_io)
            

            wav_io = convert_mp3_to_wav(mp3_bytes)
            shimmer_apq3, shimmer_apq5 = calculate_shimmer_apq(wav_io)
            

            wav_io = convert_mp3_to_wav(mp3_bytes)
            y, sr = librosa.load(wav_io, sr=None)
            hnr, nhr = calculate_hnr_nhr(y, sr)
            

            wav_io = convert_mp3_to_wav(mp3_bytes)
            y, sr = librosa.load(wav_io, sr=None)
            rpde = calculate_rpde(y, sr)
            dfa = calculate_dfa(y, sr)
        else:
            wav_io = uploaded_file.read()
            
            fo, high, low = analyze_frequencies(wav_io)
            

            jitter_percentage, abs_jitter, jitter_ddp = calculate_jitter(wav_io)
            

            mdvp_rap, mdvp_ppq, mdvp_apq = calculate_mdvp_measures(wav_io)
            

            shimmer_dda, mdvp_shimmer, mdvp_shimmer_db = calculate_shimmer(wav_io)
            

            shimmer_apq3, shimmer_apq5 = calculate_shimmer_apq(wav_io)
            

            y, sr = librosa.load(wav_io, sr=None)
            hnr, nhr = calculate_hnr_nhr(y, sr)
            

            y, sr = librosa.load(wav_io, sr=None)
            rpde = calculate_rpde(y, sr)
            
            dfa = calculate_dfa(y, sr)
        input_data = (fo, high, low, jitter_percentage, abs_jitter, mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, D2, PPE)
        
        ts = np.asarray(input_data)
        reshape_data = ts.reshape(1, -1)
        
        input_data_df = pd.DataFrame(reshape_data, columns=[
    "Fundamental Frequency", "Highest Frequency", "Lowest frequency", "Jitter Percentage", "Absulate Jitter", "MDVP RAP", "MDVP PPQ",
    "Jitter DDP", "MDVP Shimmer", "MDVP Shimmer dB", "Shimmer APQ3", "Shimmer APQ5",
    "MDVP APQ", "Shimmer DDA", "NHR", "HNR", "RPDE", "DFA", "Spread1", "Spread2", "D2", "PPE"
])
        input_data_df = input_data_df.applymap(lambda x: f"{x:.10f}")
        st.dataframe(input_data_df)

        prediction = custom_inference_function(reshape_data, loaded_model)
        col1, col2, col3 = st.columns([2.3, 2.5, 1])
        with col2:
            st.markdown(f"Prediction: {prediction}")
        
        if prediction[0] == 0:
            col1, col2, col3 = st.columns([2, 2.5, 1])
            with col2:
                st.markdown(":green[Negative, No Parkinson's Found]")
            
        else:
            col1, col2, col3 = st.columns([2, 2.5, 1])
            with col2:
                st.markdown(":red[Positive, Parkinson's Found]")
            
    