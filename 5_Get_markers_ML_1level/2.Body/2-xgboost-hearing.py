# XGBoost for Aoraki
import os
import pickle
import pandas as pd
import numpy as np
import warnings
import sys
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define helping functions
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

# Evaluate performance
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

# Set up modalities
modalities = ['hearing']

folds = range(0,5)
seed = 10
model_result = {}

# Handle command-line arguments
if len(sys.argv) > 1:
    fold = int(sys.argv[1]) % 5
    modality = int(sys.argv[1]) // 5

# Load data
base_path = '/projects/sciences/psychology/UKBiobank/brainbody/hearing-vision'
folds_path = os.path.join(base_path, f'folds/fold_{folds[fold]}')
suppl_path = os.path.join(folds_path, 'suppl')
scaling_path = os.path.join(folds_path, 'scaling')
models_path = os.path.join(folds_path, 'models')
g_pred_path = os.path.join(folds_path, 'g_pred')

for path in [folds_path, suppl_path, scaling_path, models_path, g_pred_path]:
    os.makedirs(path, exist_ok=True)
    print(f"Directory checked/created: {path}", flush=True)

# Load data files - using consistent fold variable
train_merge_all = pd.read_csv(os.path.join(suppl_path, f'{modalities[modality]}_train_feat_targ_fold_{folds[fold]}.csv'), usecols=['g', 'eid'])
test_merge_all = pd.read_csv(os.path.join(suppl_path, f'{modalities[modality]}_test_feat_targ_fold_{folds[fold]}.csv'), usecols=['g', 'eid'])

# Extract targets
g_train = train_merge_all['g']
g_test = test_merge_all['g']

# Load scaled data
features_train_scaled = pd.read_csv(os.path.join(scaling_path, f'{modalities[modality]}_train_scaled_fold_{folds[fold]}.csv'))
features_test_scaled = pd.read_csv(os.path.join(scaling_path, f'{modalities[modality]}_test_scaled_fold_{folds[fold]}.csv'))
print('Features train shape:', features_train_scaled.shape, flush=True)
print('Features test shape:', features_test_scaled.shape, flush=True)

# Initiate and run XGBoost
print('Starting XGBoost', flush=True)

parameters = {
    'booster': ['gbtree'],
    'eta': [0.1, 0.3],
    'max_depth': [3, 5],
    'min_child_weight': [1, 3],
    'subsample': [0.7, 1],
    'colsample_bytree': [0.7, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0.5, 1, 1.5],
}

xgb = XGBRegressor(random_state=seed, objective='reg:squarederror')
model = GridSearchCV(xgb, parameters, cv=5, verbose=4, n_jobs=1)

# Fit the model
print(f'Fitting XGBoost to {modalities[modality]} fold {folds[fold]}', flush=True)
model.fit(features_train_scaled, g_train)
print(f'Best params in fold {folds[fold]} = ', model.best_params_, flush=True)
print(f'Best score (neg_mean_absolute_error) in fold {folds[fold]} = ', model.best_score_, flush=True)

# Save model
model_filename = os.path.join(models_path, f'{modalities[modality]}_XGB_model_fold_{folds[fold]}.pkl')
print(f'Saving XGBoost model to {model_filename}', flush=True)
with open(model_filename, "wb") as f:
    pickle.dump(model, f)
print("Model saved successfully", flush=True)

# Predict and save
g_pred_test = save_predictions(features_test_scaled, g_test, test_merge_all['eid'], 'test', modalities[modality], fold)
g_pred_train = save_predictions(features_train_scaled, g_train, train_merge_all['eid'], 'train', modalities[modality], fold)

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
