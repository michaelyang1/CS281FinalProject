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
    
    X = data_processor.drop_top_n_features_most_correlated_with_gender(X, y, n=20)
    print(type(X), type(y))
    print(X.shape, y.shape)

    num_experiments = 10
    overall_loss_avg, male_loss_avg, female_loss_avg = 0, 0, 0
    for _ in range(num_experiments):
        # overall_loss, male_loss, female_loss = models.train_regression_model(X, y, eo=True)
        overall_loss, male_loss, female_loss = models.train_adversarial_model(X, y)
        
        overall_loss_avg += overall_loss
        male_loss_avg += male_loss
        female_loss_avg += female_loss
        
    overall_loss_avg /= num_experiments
    male_loss_avg /= num_experiments
    female_loss_avg /= num_experiments
    print(f"Overall Loss Avg: {overall_loss_avg:.4f}, Male Loss Avg: {male_loss_avg:.4f}, Female Loss Avg: {female_loss_avg:.4f}")
