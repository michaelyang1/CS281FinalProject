import audio_parser
import models
if __name__ == "__main__":
    recordings_directory = "audio_by_uttr.nosync"
    # features = audio_parser.extract_audio_features_from_recordings_directory(recordings_directory, save_to_file=False)

    features_directory = "audio_extracted_features.nosync"
    inputs = audio_parser.get_inputs_from_features_directory(features_directory)
    # print(len(inputs))
    # print(inputs[0].shape)

    targets_filename = "train_split_Depression_AVEC2017.csv"
    targets = audio_parser.get_targets_from_targets_directory(targets_filename)
    
    x, y = audio_parser.match_input_features_to_targets(inputs, targets)
    
    # models.train_baseline_regression_model(x, y)
