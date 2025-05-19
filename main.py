import audio_parser

if __name__ == "__main__":
    recordings_directory = "audio_by_uttr.nosync"
    audio_parser.extract_audio_features_from_directory(recordings_directory, save_to_file=True)
