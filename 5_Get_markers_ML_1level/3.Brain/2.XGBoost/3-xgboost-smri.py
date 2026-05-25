# XGBoost for Aoraki - Optimized Adaptive Grid
import os
import pickle
import pandas as pd
import numpy as np
import warnings
import sys
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define helping functions (unchanged)
def save_predictions(data, targets, eids, prefix, modality, fold):
    preds = model.predict(data)
    preds_df = pd.DataFrame(preds, columns=[f'g_pred_{prefix}_{modality}'])
    preds_df.to_csv(os.path.join(g_pred_path, f'{modality}_g_pred_XGB_{prefix}_fold_{folds[fold]}.csv'), index=False)
    pd.concat([eids, preds_df], axis=1).to_csv(
        os.path.join(g_pred_path, f'{modality}_g_pred_XGB_{prefix}_with_id_fold_{folds[fold]}.csv'),
        index=False
    )
    print(f"{prefix.capitalize()} predictions saved successfully", flush=True)
    return preds

def evaluate_metrics(true, pred, set_name):
    print(f"{set_name} Set Metrics:", flush=True)
    print("MSE = ", mean_squared_error(true, pred), flush=True)
    print("MAE = ", mean_absolute_error(true, pred), flush=True)
    print("R2 = ", r2_score(true, pred), flush=True)
    print("Pearson's r = ", pearsonr(true, pred)[0], flush=True)
    print("----------", flush=True)
    return {
        f'{set_name}_MSE': mean_squared_error(true, pred),
        f'{set_name}_MAE': mean_absolute_error(true, pred),
        f'{set_name}_R2': r2_score(true, pred),
        f'{set_name}_Pearson_r': pearsonr(true, pred)[0]
    }

# Enhanced adaptive parameter grid
def get_param_grid(num_features):
    """Dynamically adapt parameters based on feature count and sample size"""
    base_grid = {
        'booster': ['gbtree'],
        'objective': ['reg:squarederror'],
        'tree_method': ['hist']  # Default, will be overridden for extreme cases
    }
    
    # Ultra-low dimension (3-20 features)
    if num_features <= 20:
        return {
            **base_grid,
            'eta': [0.03, 0.05, 0.1],
            'max_depth': [4, 5, 6],          # Deeper trees to capture limited interactions
            'min_child_weight': [1, 3],       # Few samples per leaf acceptable
            'subsample': [0.9, 1.0],         # Minimal data subsampling
            'colsample_bytree': [1.0],       # Use all features (no need to subset)
            'gamma': [0],                    # No minimum loss reduction
            'reg_alpha': [0],                # No L1 regularization
            'reg_lambda': [1]                # Minimal L2
        }
    
    # Low dimension (21-100 features)
    elif num_features <= 100:
        return {
            **base_grid,
            'eta': [0.03, 0.05, 0.1],
            'max_depth': [4, 5],            # Moderate depth
            'min_child_weight': [3, 5],      # Require more samples per leaf
            'subsample': [0.8, 0.9],        # Moderate data subsampling
            'colsample_bytree': [0.7, 0.9], # Partial feature sampling
            'gamma': [0, 0.1],              # Mild node splitting control
            'reg_alpha': [0, 0.5],          # Optional L1
            'reg_lambda': [1.5, 2]          # Moderate L2
        }
    
    # Medium dimension (101-200K features)
    elif num_features <= 30000:
        return {
            **base_grid,
            'eta': [0.01, 0.05],
            'max_depth': [3, 4],            # Shallow trees
            'min_child_weight': [5, 10],    # Strict leaf requirements
            'subsample': [0.6, 0.8],       # Aggressive data subsampling
            'colsample_bytree': [0.5, 0.7], # Heavy feature sampling
            'gamma': [0.3, 0.5],           # Strong node splitting control
            'reg_alpha': [0.5, 1],         # Strong L1
            'reg_lambda': [2, 3],          # Strong L2
            'max_bin': [256]               # Faster training
        }
    
    # Extreme dimension (200K-500K+ features)
    else:
        return {
            **base_grid,
            'eta': [0.01, 0.05],           # Very small learning rate
            'tree_method': ['hist'],    # GPU acceleration mandatory
            'max_depth': [2, 3],           # Very shallow trees
            'min_child_weight': [10, 20],  # Large min samples per leaf
            'subsample': [0.5, 0.7],       # Heavy data subsampling
            'colsample_bytree': [0.3, 0.5], # Sample <<1% of features per tree
            'colsample_bylevel': [0.2, 0.3], # Further subsample per level
            'gamma': [0.5, 1],               # Stronger node splitting control
            'reg_alpha': [1, 2],           # Aggressive L1 (induces sparsity)
            'reg_lambda': [3, 5],          # Strong L2 regularization
            'max_bin': [128, 256],         # Fewer bins for speed
            'n_estimators': [1000, 1500, 2000],  # More trees due to smaller eta
            #'single_precision_histogram': [True], # Faster GPU histograms
            #'predictor': ['gpu_predictor'] # GPU for inference
        }


# Set up modalities
modalities = [
'struct_fast',
'struct_sub_first',
'struct_fs_aseg_mean_intensity',
'struct_fs_aseg_volume',
'struct_ba_exvivo_area', 
'struct_ba_exvivo_mean_thickness',
'struct_ba_exvivo_volume',
'struct_a2009s_area','struct_a2009s_mean_thickness','struct_a2009s_volume',
'struct_dkt_area', 'struct_dkt_mean_thickness', 'struct_dkt_volume',
'struct_desikan_gw', 'struct_desikan_pial',
'struct_desikan_white_area', 'struct_desikan_white_mean_thickness', 'struct_desikan_white_volume',
'struct_subsegmentation',
'add_t1',
'add_t2'
] #21

folds = range(0,5)
seed = 10
model_result = {}

# Handle command-line arguments
if len(sys.argv) > 1:
    fold = int(sys.argv[1]) % 5
    modality = int(sys.argv[1]) // 5

# Load data
base_path = '/mnt/UK_BB/brainbody/brain'
folds_path = os.path.join(base_path, f'folds/fold_{folds[fold]}')
suppl_path = os.path.join(folds_path, 'suppl')
scaling_path = os.path.join(folds_path, 'scaling')
models_path = os.path.join(folds_path, 'models')
g_pred_path = os.path.join(folds_path, 'g_pred')

for path in [folds_path, suppl_path, scaling_path, models_path, g_pred_path]:
    os.makedirs(path, exist_ok=True)
    print(f"Directory checked/created: {path}", flush=True)

# Upload targets and eid
targets_train = pd.read_csv(os.path.join(suppl_path, f'{modalities[modality]}_train_targets_fold_{fold}.csv'), usecols = ['eid', 'g'])
targets_test = pd.read_csv(os.path.join(suppl_path, f'{modalities[modality]}_test_targets_fold_{fold}.csv'), usecols = ['eid', 'g'])

g_train = targets_train['g']
g_test = targets_test['g']

eid_train = targets_train['eid']
eid_test = targets_test['eid']

# Load scaled data
features_train_deconf = pd.read_csv(os.path.join(scaling_path, f'{modalities[modality]}_train_deconf_fold_{folds[fold]}.csv'))
features_test_deconf = pd.read_csv(os.path.join(scaling_path, f'{modalities[modality]}_test_deconf_fold_{folds[fold]}.csv'))
print('Features train shape:', features_train_deconf.shape, flush=True)
print('Features test shape:', features_test_deconf.shape, flush=True)

# Initiate and run XGBoost
print('Starting XGBoost', flush=True)

# Get adaptive parameter grid
num_features = features_train_deconf.shape[1]
parameters = get_param_grid(num_features)
print(f'Using adaptive grid for {num_features} features:', parameters, flush=True)

xgb = XGBRegressor(
    random_state=seed,
    objective='reg:squarederror'
)

# Choose search strategy based on feature count
if num_features > 40000:  # Use RandomizedSearchCV for large feature sets
    print(f"Large feature set ({num_features} features) detected. Using RandomizedSearchCV", flush=True)
    model = RandomizedSearchCV(
        xgb,  # Using your pre-initialized model
        parameters,
        n_iter=5,  # Test 5 random combinations
        cv=5,       # Fewer folds for speed
        verbose=4,
        n_jobs=1,   # Matches your original setting
        random_state=seed
    )
else:  # Use GridSearchCV for smaller feature sets
    print(f"Using GridSearchCV for {num_features} features", flush=True)
    model = GridSearchCV(
        xgb,  # Using your pre-initialized model
        parameters,
        cv=5,
        verbose=4,
        n_jobs=1    # Matches your original setting
    )

# Fit with early stopping
print(f'Fitting XGBoost to {modalities[modality]} fold {folds[fold]}', flush=True)
model.fit(
    features_train_deconf, 
    g_train,
    verbose=True,
)

print(f'Best params in fold {folds[fold]} = ', model.best_params_, flush=True)
print(f'Best score (neg_mean_absolute_error) in fold {folds[fold]} = ', model.best_score_, flush=True)

# Save model
model_filename = os.path.join(models_path, f'{modalities[modality]}_XGB_model_fold_{folds[fold]}.pkl')
print(f'Saving XGBoost model to {model_filename}', flush=True)
with open(model_filename, "wb") as f:
    pickle.dump(model, f)
print("Model saved successfully", flush=True)

# Predict and save
g_pred_test = save_predictions(features_test_deconf, g_test, eid_test, 'test', modalities[modality], fold)
g_pred_train = save_predictions(features_train_deconf, g_train, eid_train, 'train', modalities[modality], fold)

# Evaluate and store results
print(f"Fold = {folds[fold]}", flush=True)
print("----------", flush=True)
test_metrics = evaluate_metrics(g_test, g_pred_test, "Test")
train_metrics = evaluate_metrics(g_train, g_pred_train, "Train")

# Store results
model_result.update({
    'Fold': fold,
    'Modality': modalities[modality],
    'Best_params': str(model.best_params_)
})
model_result.update(test_metrics)
model_result.update(train_metrics)

# Save results
results_filename = os.path.join(models_path, f'{modalities[modality]}_XGB_result_fold_{folds[fold]}.csv')
results_df = pd.DataFrame([model_result])
results_df.to_csv(
    results_filename,
    mode='a',
    index=False,
    header=not os.path.exists(results_filename)
)
print(f"Results saved to {results_filename}", flush=True)
