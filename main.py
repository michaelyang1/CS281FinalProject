import audio_parser
import models
import data_processor

if __name__ == "__main__":
    recordings_directory = "audio_by_uttr.nosync"

    features_directory = "audio_extracted_features.nosync"
    inputs = audio_parser.get_inputs_from_features_directory(features_directory)

    targets_filename = "train_split_Depression_AVEC2017.csv"
    targets = audio_parser.get_targets_from_targets_directory(targets_filename)
    
    X, y = audio_parser.match_input_features_to_targets(inputs, targets)
    
    X = data_processor.get_top_n_features_most_correlated_with_gender(X, y)
    print(type(X), type(y))
    print(X.shape, y.shape)

    models.train_regression_model(X, y)
