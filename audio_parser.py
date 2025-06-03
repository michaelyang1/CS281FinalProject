import audiofile
import opensmile
import os
import pandas as pd
import numpy as np
def extract_audio_features_from_recordings_directory(recordings_directory, save_to_file=False):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPS,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    features_list = []
    for recording_filename in os.listdir(recordings_directory):
        if recording_filename.endswith(".wav"):
            # get signals and sampling rate of audio recording using audiofile library
            signals, sampling_rate = audiofile.read(os.path.join(recordings_directory, recording_filename))
            
            # extract features from audio recording using opensmile
            features = smile.process_signal(signals, sampling_rate)
            features_list.append(features)
            
            # save features to file
            if save_to_file:
                features_directory_prefix = "audio_extracted_features.nosync/"
                features_filename = f"{features_directory_prefix}{recording_filename[:-3]}csv"
                features.to_csv(features_filename)
    return features_list

def get_inputs_from_features_directory(features_directory):
    inputs_dict = {}
    for filename in os.listdir(features_directory):
        speaker_id = filename[:7]
        feature_sample_raw = pd.read_csv(os.path.join(features_directory, filename))
        feature_sample = feature_sample_raw.iloc[:, 2:].astype(float) # skip start and end times because those aren't features
        if np.isinf(feature_sample.values).any() or np.isnan(feature_sample.values).any():
            print(f"Warning: {filename} contains inf or nan values; skipping file")
            continue
        
        if speaker_id not in inputs_dict:
            inputs_dict[speaker_id] = [feature_sample]
        else:
            inputs_dict[speaker_id].append(feature_sample)
    return inputs_dict

def get_targets_from_targets_directory(filename):
    target_data = pd.read_csv(filename)
    targets_dict = {}
    for _, row in target_data.iterrows():
        speaker_id = f"spk_{int(row['Participant_ID'])}"        
        phq8_binary = row['PHQ8_Binary']
        phq8_score = row['PHQ8_Score']
        gender = row['Gender']
        
        # phq8_binary is 1 if phq8_score >= 10 else 0
        # print(speaker_id, phq8_binary, phq8_score)
        targets_dict[speaker_id] = pd.DataFrame({'PHQ8_Score': [phq8_score], 'Gender': [gender]})
    return targets_dict

def match_input_features_to_targets(inputs_dict, targets_dict):
    # we need this function because there are ~150 input features for each target
    # ex: there are 150 audio recordings for a specific speaker, but that speaker only produces one depression outcome (i.e. many to one)
    X = []
    y = []
    for speaker_id in inputs_dict.keys():
        speaker_inputs = inputs_dict[speaker_id] # list of many dataframes
        speaker_target = targets_dict[speaker_id] # list of single dataframe
        
        for input in speaker_inputs:
            X.append(input)
            y.append(speaker_target)
    
    X = pd.concat(X, axis=0)
    y = pd.concat(y, axis=0)
    return X, y
