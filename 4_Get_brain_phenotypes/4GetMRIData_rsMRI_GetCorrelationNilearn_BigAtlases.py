import pandas as pd
import pickle
import gc
import numpy as np
from nilearn.connectome import ConnectivityMeasure
import os
import sys
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ========================
# CONFIGURATION
# ========================
connectomes_dir = '/mnt/auto-hcs/sci-psy-narun/IBu/UK_BB/brainbody/brain/data/rsMRI/connectomes'

def process_modality_part2(modality):
    print(f'Processing {modality} - Part 2: Correlation Computation', flush=True)
    
    # Load keys
    keys_df = pd.read_csv(os.path.join(connectomes_dir, f'nilearn/{modality}_id.csv'))
    keys = keys_df['eid'].tolist()
    
    # Initialize connectivity measures
    print('Initializing connectivity measures', flush=True)
    correlation_measure = ConnectivityMeasure(
        kind='correlation',
        standardize="zscore_sample",
        vectorize=True,
        discard_diagonal=True
    )
    
    partial_correlation_measure = ConnectivityMeasure(
        kind='partial correlation',
        standardize="zscore_sample",
        vectorize=True,
        discard_diagonal=True
    )
    
    # Initialize result containers
    full_correlation = []
    partial_correlation = []
    nan_indices = []
    valid_ids = []
    
    # Process each subject
    for i, key in enumerate(tqdm(keys)):
        try:
            # Load pre-saved transposed array
            subject_path = os.path.join(connectomes_dir, 'nilearn', f'{modality}_array_transposed.pkl')
            with open(subject_path, "rb") as f:
                array = pickle.load(f)

            if np.isnan(array).any():
                print(f"Skipping {key} due to NaN values", flush=True)
                nan_indices.append(i)
                continue
                
            # Compute both correlation and partial correlation
            corr_matrix = correlation_measure.fit_transform([array])
            partial_matrix = partial_correlation_measure.fit_transform([array])
            
            # Apply Fisher Z-transform
            full_correlation.append(np.arctanh(corr_matrix))
            partial_correlation.append(np.arctanh(partial_matrix))
            valid_ids.append(str(key))
            
            # Clean up
            del array, corr_matrix, partial_matrix
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {key}: {str(e)}", flush=True)
            nan_indices.append(i)
    
    # Save results
    if full_correlation:
        # Convert to DataFrames with eid
        full_corr_df = pd.DataFrame(
            np.squeeze(full_correlation),
            index=valid_ids
        )
        partial_corr_df = pd.DataFrame(
            np.squeeze(partial_correlation),
            index=valid_ids
        )
        
        # Save as CSV
        full_corr_df.to_csv(
            os.path.join(connectomes_dir, f'nilearn/full_correlation_{modality}.csv'),
            index_label='eid'
        )
        partial_corr_df.to_csv(
            os.path.join(connectomes_dir, f'nilearn/partial_correlation_{modality}.csv'),
            index_label='eid'
        )
        
        # Also save as pickle
        #with open(os.path.join(connectomes_dir, f'nilearn/full_correlation_{modality}.pkl'), 'wb') as f:
            #pickle.dump(full_corr_df, f, protocol=pickle.HIGHEST_PROTOCOL)
        #with open(os.path.join(connectomes_dir, f'nilearn/partial_correlation_{modality}.pkl'), 'wb') as f:
            #pickle.dump(partial_corr_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save NaN indices
    if nan_indices:
        nan_ids = [str(keys[i]) for i in nan_indices]
        pd.DataFrame(nan_ids, columns=['eid']).to_csv(
            os.path.join(connectomes_dir, f'nilearn/{modality}_nan_id.csv'),
            index=False
        )
    
    print(f"Completed Part 2 for {modality}. Skipped {len(nan_indices)} participants.", flush=True)
    gc.collect()

if __name__ == "__main__":
    modalities = [
        #'aparc_a2009s_Tian_S1',
        #'aparc_Tian_S1',
        #'Glasser_Tian_S1',
        #'Glasser_Tian_S4',
        #'Schaefer7n200p_Tian_S1',
        'Schaefer7n500p_Tian_S4',
        'Schaefer7n1000p_Tian_S4'
    ]
    
    # SLURM array support
    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])
        modality = modalities[task_id]
        process_modality_part2(modality)
    else:
        for modality in modalities:
            process_modality_part2(modality)
            gc.collect()
