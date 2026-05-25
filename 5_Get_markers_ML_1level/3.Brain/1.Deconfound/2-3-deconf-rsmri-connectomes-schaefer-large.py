# Memory-efficient deconfounding
import os
import pandas as pd
import numpy as np
import sys
import gc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Configuration
BASE_PATH = "/mnt/UK_BB/brainbody"
CONNECTOMES_DIR = os.path.join(BASE_PATH, "brain/data/rsMRI/connectomes/nilearn")

# Define modalities
rs_modalities = [
'full_correlation_Schaefer7n500p_Tian_S4',
'partial_correlation_Schaefer7n500p_Tian_S4'
]

def upload_file(modality_name):
    """Find the correct file path for a given modality"""
    file_path = os.path.join(CONNECTOMES_DIR, f"{modality_name}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file at: {file_path}")
    
    return file_path

def load_data_with_memory_efficiency(file_path, usecols=None):
    """Load CSV file with memory-efficient settings"""
    return pd.read_csv(file_path, usecols=usecols, engine='c')

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
    """Process a single modality-fold combination with memory optimization"""
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
        # Load only necessary columns
        g_path = os.path.join(fold_paths['cognition'], f'folds/fold_{fold}/g')
        g_train = load_data_with_memory_efficiency(
            os.path.join(g_path, f'g_train_with_id_{fold}.csv'),
            usecols=['eid', 'g']
        )
        g_test = load_data_with_memory_efficiency(
            os.path.join(g_path, f'g_test_with_id_{fold}.csv'),
            usecols=['eid', 'g']
        )
        
        # Load features and confounds with minimal memory footprint
        features = load_data_with_memory_efficiency(upload_file(modality))
        confounds = load_data_with_memory_efficiency(
            os.path.join(BASE_PATH, 'brain/data/rsMRI/rsmri_conf.csv')
        )
        
        # Get column lists before merging
        feature_cols = [col for col in features.columns if col != 'eid']
        confound_cols = [col for col in confounds.columns if col != 'eid']
        
        # Memory-efficient merging - process one dataset at a time
        print("Merging training data...")
        train_data = features[['eid'] + feature_cols].merge(
            g_train, on='eid', how='inner'
        ).merge(
            confounds[['eid'] + confound_cols], on='eid', how='inner'
        )
        
        print("Merging test data...")
        test_data = features[['eid'] + feature_cols].merge(
            g_test, on='eid', how='inner'
        ).merge(
            confounds[['eid'] + confound_cols], on='eid', how='inner'
        )
        
        # Free memory
        del features, confounds, g_train, g_test
        gc.collect()
        
        # Save targets
        train_data[['eid', 'g']].to_csv(
            os.path.join(fold_paths['suppl'], f'{modality}_train_targets_fold_{fold}.csv'),
            index=False
        )
        test_data[['eid', 'g']].to_csv(
            os.path.join(fold_paths['suppl'], f'{modality}_test_targets_fold_{fold}.csv'),
            index=False
        )
        
        # Process features in numpy arrays to reduce memory overhead
        print("Running confound regression...")
        features_train, features_test = confound_regressor(
            train_data[feature_cols].values,
            test_data[feature_cols].values,
            train_data[confound_cols].values,
            test_data[confound_cols].values
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
        
        # Final cleanup
        del train_data, test_data, features_train, features_test
        gc.collect()
        
    except Exception as e:
        print(f"Error processing {modality} fold {fold}: {str(e)}")
        raise

if __name__ == "__main__":
    # Get job index from SLURM
    try:
        job_idx = int(sys.argv[1])
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
