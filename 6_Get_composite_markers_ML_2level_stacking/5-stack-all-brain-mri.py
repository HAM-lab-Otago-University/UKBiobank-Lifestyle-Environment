# Random forest grid for stacking: rs
import csv
import os
import pickle
import pandas as pd
import numpy as np
import sys
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Configuration
algorithms = ['rf']
folds = ["0", "1", "2", "3", "4"]
modality_name = 'allmri'
seed = 42 #10

if len(sys.argv) > 1:
    fold = int(sys.argv[1]) % 5
    algorithm = int(sys.argv[1]) // 5

base_path = '/mnt/UK_BB/brainbody'
stacking_path = os.path.join(base_path, 'stacking', 'brain', modality_name)
cognition_path = os.path.join(base_path, 'cognition')
folds_path = os.path.join(stacking_path, 'folds', f'fold_{folds[fold]}')
suppl_path = os.path.join(folds_path, 'suppl')
scaling_path = os.path.join(folds_path, 'scaling')
models_path = os.path.join(folds_path, 'models')
g_pred_path = os.path.join(folds_path, 'g_pred')

# Create directories
for path in [folds_path, suppl_path, scaling_path, models_path, g_pred_path]:
    os.makedirs(path, exist_ok=True)

def load_data(fold, data_type):
    """Load pre-matched features and target data."""
    # Set file path - now correctly pointing to different subfolders for train/test
    if data_type == 'train':
        data_path = os.path.join(stacking_path, 'features_train_level1_stacked_inner', 
                               f'features_train_level1_inner_g_matched_fold_{fold}.csv')
    elif data_type == 'test':
        data_path = os.path.join(stacking_path, 'features_test_level1_stacked_inner',
                               f'features_test_level1_inner_g_matched_fold_{fold}.csv')
    
    # Verify path exists before loading
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    # Rest of the function remains the same...
    df = pd.read_csv(data_path)
    print(f"\nLoading {data_type} data for fold {fold} from: {data_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    
    target_values = df['g'].values.reshape(-1, 1)
    target_ids = df['eid']
    features = df.drop(columns=['eid', 'g'])
    features.columns = features.columns.str.replace(f'_{data_type}_', '')
    
    return features, target_values, target_ids

# Load train and test data with updated paths
features_train, target_real_train, target_real_train_id = load_data(folds[fold], 'train')
features_test, target_real_test, target_real_test_id = load_data(folds[fold], 'test')

# Scaling
scaler_features = StandardScaler()
scaler_target = StandardScaler()

target_real_train = scaler_target.fit_transform(target_real_train)
target_real_test = scaler_target.transform(target_real_test)

features_train = scaler_features.fit_transform(features_train)
features_test = scaler_features.transform(features_test)

# Save scalers
with open(os.path.join(suppl_path, f'{modality_name}_scaler_features_fold_{folds[fold]}.pkl'), 'wb') as f:
    pickle.dump(scaler_features, f)
    
with open(os.path.join(suppl_path, f'{modality_name}_scaler_target_fold_{folds[fold]}.pkl'), 'wb') as f:
    pickle.dump(scaler_target, f)

# Model training
algorithm_performance = {}

if algorithms[algorithm] == 'xgb':
    param = {
        'booster': ['gbtree'],
        'eta': [0.01, 0.05, 0.1],
        'max_depth': [1, 2, 3],
        'subsample': [0.8, 1],
        'reg_lambda': [0, 0.1, 0.5],
        'reg_alpha': [0, 0.01, 0.1],
        'min_child_weight': [1, 3],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1]
    }
    regressor = xgb.XGBRegressor(
    random_state=seed,
    objective='reg:squarederror',
    )

elif algorithms[algorithm] =='rf':
    print('Started RF')
    param ={'n_estimators': [5000],
                'max_depth':[1,2,3,4,5,6],
                'max_features':['sqrt','log2']}
    regressor = RandomForestRegressor()

model = GridSearchCV(regressor, param, cv=5, verbose=4, n_jobs=1)
model.fit(features_train, target_real_train.ravel())

# Save model
with open(os.path.join(models_path, f'{modality_name}_stacked_{algorithms[algorithm]}_model_fold_{folds[fold]}.pkl'), 'wb') as f:
    pickle.dump(model, f)

# Predictions
target_pred_train = model.predict(features_train)
target_pred_test = model.predict(features_test)

# Save predictions
pd.DataFrame({'eid': target_real_train_id, 'g_pred_stack_train': target_pred_train}).to_csv(
    os.path.join(g_pred_path, f'{modality_name}_target_pred_2nd_level_{algorithms[algorithm]}_train_fold_{folds[fold]}.csv'),
    index=False
)

pd.DataFrame({'eid': target_real_test_id, 'g_pred_stack_test': target_pred_test}).to_csv(
    os.path.join(g_pred_path, f'{modality_name}_target_pred_2nd_level_{algorithms[algorithm]}_test_fold_{folds[fold]}.csv'),
    index=False
)

# Store results
algorithm_performance = {
    'Algorithm': algorithms[algorithm],
    'Fold': folds[fold],
    'Best_params': str(model.best_params_),
    'Test_MSE': mean_squared_error(target_real_test, target_pred_test),
    'Test_MAE': mean_absolute_error(target_real_test, target_pred_test),
    'Test_R2': r2_score(target_real_test, target_pred_test),
    'Test_Pearson_r': pearsonr(target_real_test.squeeze(), target_pred_test.squeeze())[0],
    'Train_MSE': mean_squared_error(target_real_train, target_pred_train),
    'Train_MAE': mean_absolute_error(target_real_train, target_pred_train),
    'Train_R2': r2_score(target_real_train, target_pred_train),
    'Train_Pearson_r': pearsonr(target_real_train.squeeze(), target_pred_train.squeeze())[0]
}

# Write results
results_file = os.path.join(models_path, f'{modality_name}_{algorithms[algorithm]}_stacked_result_fold_{folds[fold]}.csv')
write_header = not os.path.exists(results_file)

with open(results_file, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=algorithm_performance.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(algorithm_performance)
