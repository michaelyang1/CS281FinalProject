import audiofile
import opensmile
import os
import pandas as pd

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
    inputs = []
    for filename in os.listdir(features_directory):
        feature_sample = pd.read_csv(os.path.join(features_directory, filename))
        inputs.append(feature_sample)
        print(filename)
    return inputs

def get_targets_from_targets_directory(filename):
    targets = pd.read_csv(filename)
    return targets

def match_features_to_targets(features, targets):
    pass
