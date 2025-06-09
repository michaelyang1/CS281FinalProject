import scipy.stats as stats
import numpy as np

def drop_top_n_features_most_correlated_with_gender(X, y, n=0):
    # get correlation between each feature and gender
    correlations = []
    for feature in X.columns:
        correlation = stats.pearsonr(X[feature], y['Gender'])
        correlations.append((feature, correlation))
    
    # sort by correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    for i, (feature, correlation) in enumerate(correlations[:n]):
        print(f"{i}: {feature}, {correlation}")
        
    # remove the feature columns from X in top n
    X = X.drop(columns=[feature for feature, _ in correlations[:n]])
    return X
