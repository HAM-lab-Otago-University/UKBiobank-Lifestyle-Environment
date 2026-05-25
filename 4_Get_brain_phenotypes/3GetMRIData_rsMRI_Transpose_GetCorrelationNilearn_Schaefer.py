# Get full and partial correlations in parallel: Schaeffer 500
import pandas as pd
import pickle
import gc
import numpy as np
from nilearn.connectome import ConnectivityMeasure
import os
import sys
import warnings
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback
warnings.filterwarnings('ignore')

# ========================
# CONFIGURATION
# ========================
connectomes_dir = '/mnt/auto-hcs/sci-psy-narun/IBu/UK_BB/brainbody/brain/data/rsMRI/connectomes'

def process_subject(args):
    """Process a single subject in parallel"""
    modality, key, i = args
    try:
        # Load pre-saved transposed array
        subject_path = os.path.join(connectomes_dir, 'nilearn', f'{modality}_array_transposed.pkl')
        with open(subject_path, "rb") as f:
            # First item is count, then arrays
            count = pickle.load(f)
            if i >= count:
                return (key, None, None, "Index out of bounds")
            
            # Skip to the correct array
            for idx in range(i + 1):  # +1 because we want to include current index
                array = pickle.load(f)
                if idx != i:
                    continue  # Skip until we reach our target index

        # Handle case where array is not a numpy array
        if not isinstance(array, np.ndarray):
            if isinstance(array, (int, float)):
                return (key, None, None, f"Expected array, got scalar value: {array}")
            else:
                return (key, None, None, f"Unexpected data type: {type(array)}")

        # Shape validation
        if array.ndim != 2:
            return (key, None, None, f"Invalid array dimensions: {array.ndim}")

        if np.isnan(array).any():
            return (key, None, None, "NaN values")
            
        # Initialize measures
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

        # Compute correlations
        corr_matrix = correlation_measure.fit_transform([array])
        partial_matrix = partial_correlation_measure.fit_transform([array])
        
        # Apply Fisher Z-transform
        return (
            str(key), 
            np.arctanh(corr_matrix).squeeze(), 
            np.arctanh(partial_matrix).squeeze(), 
            None
        )
        
    except Exception as e:
        return (str(key), None, None, f"{str(e)}\n{traceback.format_exc()}")

def process_modality_part2(modality):
    print(f'Processing {modality} - Part 2: Correlation Computation', flush=True)
    
    # Load keys
    keys_df = pd.read_csv(os.path.join(connectomes_dir, f'nilearn/{modality}_id.csv'))
    keys = keys_df['eid'].tolist()
    
    # Get number of available CPUs
    n_processes = min(cpu_count(), len(keys))
    print(f"Using {n_processes} parallel processes", flush=True)

    # Prepare arguments for parallel processing
    args = [(modality, key, i) for i, key in enumerate(keys)]
    
    # Initialize result containers
    full_correlation = []
    partial_correlation = []
    valid_ids = []
    nan_indices = []
    error_messages = []

    # Process in parallel
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_subject, args),
            total=len(keys),
            desc="Processing subjects"
        ))

    # Process results
    for i, (key, corr, partial, error) in enumerate(results):
        if error:
            if "NaN values" in error:
                nan_indices.append(i)
            else:
                print(f"Error processing {key}: {error.splitlines()[0]}", flush=True)
                nan_indices.append(i)
            continue
            
        if corr is not None and partial is not None:
            full_correlation.append(corr)
            partial_correlation.append(partial)
            valid_ids.append(key)

    # Save results
    if full_correlation:
        # Stack results ensuring consistent shapes
        full_corr_stack = np.vstack(full_correlation)
        partial_corr_stack = np.vstack(partial_correlation)
        
        # Convert to DataFrames
        full_corr_df = pd.DataFrame(full_corr_stack, index=valid_ids)
        partial_corr_df = pd.DataFrame(partial_corr_stack, index=valid_ids)
        
        # Save as CSV
        full_corr_df.to_csv(
            os.path.join(connectomes_dir, f'nilearn/full_correlation_{modality}.csv'),
            index_label='eid'
        )
        partial_corr_df.to_csv(
            os.path.join(connectomes_dir, f'nilearn/partial_correlation_{modality}.csv'),
            index_label='eid'
        )

    # Save NaN/error indices
    if nan_indices:
        nan_ids = [str(keys[i]) for i in nan_indices]
        pd.DataFrame(nan_ids, columns=['eid']).to_csv(
            os.path.join(connectomes_dir, f'nilearn/{modality}_nan_id.csv'),
            index=False
        )
    
    print(f"Completed Part 2 for {modality}. Success: {len(valid_ids)}, Skipped: {len(nan_indices)}", flush=True)
    gc.collect()

if __name__ == "__main__":
    # Ensure proper multiprocessing on Windows
    if sys.platform.startswith('win'):
        from multiprocessing import freeze_support
        freeze_support()

    modalities = [
        'Schaefer7n500p_Tian_S4'
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
