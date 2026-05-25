# De-looped script to Aoraki
import os
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Configuration
BASE_PATH = "/mnt/UK_BB/brainbody"
CONNECTOMES_DIR = os.path.join(BASE_PATH, "brain/data/rsMRI/connectomes")

# Define modalities
rs_modalities = [
'full_correlation_aparc_a2009s_Tian_S1',
'full_correlation_aparc_Tian_S1',
'full_correlation_Glasser_Tian_S1',
'full_correlation_Glasser_Tian_S4',
'full_correlation_Schaefer7n200p_Tian_S1',
'full_correlation_Schaefer7n500p_Tian_S4',
'full_correlation_Schaefer7n1000p_Tian_S4',
'partial_correlation_aparc_a2009s_Tian_S1',
'partial_correlation_aparc_Tian_S1',
'partial_correlation_Glasser_Tian_S1',
'partial_correlation_Glasser_Tian_S4',
'partial_correlation_Schaefer7n200p_Tian_S1',
'partial_correlation_Schaefer7n500p_Tian_S4',
'partial_correlation_Schaefer7n1000p_Tian_S4',
]

# Define confound regressor
def confound_regressor(features_train, features_test, confounds_train, confounds_test):
    """
    Regress out confounds from features using a linear model.
    Both features and confounds are scaled.

    Parameters:
        features_train (array-like): Training features (n_samples, n_features).
        features_test (array-like): Test features (n_samples, n_features).
        confounds_train (array-like): Training confounds (n_samples, n_confounds).
        confounds_test (array-like): Test confounds (n_samples, n_confounds).

    Returns:
        features_train_res (array-like): Confound-corrected training features (scaled).
        features_test_res (array-like): Confound-corrected test features (scaled).
    """
    # Scale features (train and test sets)
    scaler_features = StandardScaler()
    features_train = scaler_features.fit_transform(features_train)
    features_test = scaler_features.transform(features_test)
        
    # Scale confounds (train and test sets)
    scaler_confounds = StandardScaler()
    confounds_train = scaler_confounds.fit_transform(confounds_train)
    confounds_test = scaler_confounds.transform(confounds_test)
        
    # Initialize & fit the linear regression model
    model = LinearRegression(fit_intercept=False)
    model.fit(confounds_train, features_train)
    
    # Predict confound effects on training and test sets
    features_train_pred = model.predict(confounds_train)
    features_test_pred = model.predict(confounds_test)
    
    # Compute residuals (confound-corrected features)
    features_train_res = features_train - features_train_pred
    features_test_res = features_test - features_test_pred

    if confounds_train.shape[1] > 0:  # If there are confounds
        corrs = [
            np.corrcoef(confounds_train[:,i], features_train_res[:,0])[0,1] 
            for i in range(confounds_train.shape[1])
        ]
        print(f"Mean |correlation| between confounds and residuals: {np.mean(np.abs(corrs)):.3f}")

    return features_train_res, features_test_res

def deconfound_rs(modality, fold):
    """Process a single modality-fold combination"""
    print(f"\n=== Processing {modality}, Fold {fold} ===")
    
    # Path setup
    fold_paths = {
        'main': os.path.join(BASE_PATH, f'brain/folds/fold_{fold}'),
        'suppl': os.path.join(BASE_PATH, f'brain/folds/fold_{fold}/suppl'),
        'scaling': os.path.join(BASE_PATH, f'brain/folds/fold_{fold}/scaling'),
        'cognition': os.path.join(BASE_PATH, 'cognition')
    }
    
    # Create directories
    for path in fold_paths.values():
        os.makedirs(path, exist_ok=True)

    try:
        # Load data
        g_train = pd.read_csv(os.path.join(fold_paths['cognition'], f'folds/fold_{fold}/g/g_train_with_id_{fold}.csv'))
        g_test = pd.read_csv(os.path.join(fold_paths['cognition'], f'folds/fold_{fold}/g/g_test_with_id_{fold}.csv'))
        
        features = pd.read_csv(os.path.join(CONNECTOMES_DIR, f'nilearn/{modality}.csv'))

        confounds = pd.read_csv(os.path.join(BASE_PATH, 'brain/data/rsMRI/rsmri_conf.csv'))
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

    # Data processing
    feature_cols = [col for col in features.columns if col != 'eid']
    confound_cols = [col for col in confounds.columns if col != 'eid']
    
    train_data = features.merge(g_train, on='eid').merge(confounds, on='eid')
    test_data = features.merge(g_test, on='eid').merge(confounds, on='eid')
    
    # Save raw data
    train_data.to_csv(os.path.join(fold_paths['suppl'], f'{modality}_train_feat_targ_conf_fold_{fold}.csv'), index=False)
    test_data.to_csv(os.path.join(fold_paths['suppl'], f'{modality}_test_feat_targ_conf_fold_{fold}.csv'), index=False)

    train_data[['eid', 'g']].to_csv(os.path.join(fold_paths['suppl'], f'{modality}_train_targets_fold_{fold}.csv'), index=False)
    test_data[['eid', 'g']].to_csv(os.path.join(fold_paths['suppl'], f'{modality}_test_targets_fold_{fold}.csv'), index=False)

    # Confound regression
    features_train, features_test = confound_regressor(
        train_data[feature_cols],
        test_data[feature_cols],
        train_data[confound_cols],
        test_data[confound_cols]
    )
    
    # Save results
    pd.DataFrame(features_train, columns=feature_cols).to_csv(
        os.path.join(fold_paths['scaling'], f'{modality}_train_deconf_fold_{fold}.csv'), 
        index=False
    )
    pd.DataFrame(features_test, columns=feature_cols).to_csv(
        os.path.join(fold_paths['scaling'], f'{modality}_test_deconf_fold_{fold}.csv'), 
        index=False
    )

if __name__ == "__main__":
    # Get job index from SLURM
    try:
        job_idx = int(sys.argv[1])  # Single integer argument
    except (IndexError, ValueError):
        print("Error: Please provide a job index as argument")
        sys.exit(1)
    
    # Create all possible job combinations
    folds = range(5)
    jobs = [(mod, fold) for mod in rs_modalities for fold in folds]
    
    # Validate job index
    if job_idx >= len(jobs):
        print(f"Error: Job index {job_idx} exceeds maximum {len(jobs)-1}")
        sys.exit(1)
    
    # Get this job's parameters
    modality, fold = jobs[job_idx]
    
    try:
        deconfound_rs(modality, fold)
        print(f"Successfully completed {modality} fold {fold}")
    except Exception as e:
        print(f"Failed processing {modality} fold {fold}: {str(e)}")
        sys.exit(1)
