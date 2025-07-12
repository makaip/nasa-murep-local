import pandas as pd
import glob
import os
import re

def load_cdom_412nm_data(spectra_dir="spectra"):
    """
    Load CDOM spectral data from CSV files matching pattern spectra_WC_GOM{year}{season}_*.csv
    and extract data from the wavelength column closest to 412nm.
    
    Args:
        spectra_dir (str): Directory containing the spectra CSV files
    
    Returns:
        pd.DataFrame: Combined dataframe with all sample data at wavelength closest to 412nm
    """
    # Load metadata
    metadata_file = os.path.join(spectra_dir, "meta_data(in).csv")
    metadata_df = None
    if os.path.exists(metadata_file):
        try:
            metadata_df = pd.read_csv(metadata_file)
            # Rename Sample column to match our sample_id column
            if 'Sample' in metadata_df.columns:
                metadata_df = metadata_df.rename(columns={'Sample': 'sample_id'})
        except Exception as e:
            print(f"Warning: Could not load metadata file {metadata_file}: {e}")
    
    # Match any file starting with 'spectra_WC_GOM' and ending with .csv
    csv_files = glob.glob(os.path.join(spectra_dir, "sprectra_pchip_baseline_corrected_WC_GOM*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No matching CSV files found in '{spectra_dir}'. "
                                f"Make sure files like 'spectra_baseline_corrected_WC_GOM2021_St.MK_dil2X.csv' exist.")

    if not csv_files:
        raise FileNotFoundError(f"No matching CSV files found in {spectra_dir}")
    
    all_data = []
    
    for csv_file in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Find column closest to 412nm
            # Assume first row contains wavelength values as column headers
            wavelength_cols = []
            for col in df.columns:
                try:
                    wavelength = float(col)
                    wavelength_cols.append((abs(wavelength - 412), col, wavelength))
                except ValueError:
                    continue
            
            if not wavelength_cols:
                print(f"Warning: No numeric wavelength columns found in {csv_file}")
                continue
                
            # Sort by distance from 412nm and get closest
            wavelength_cols.sort()
            closest_col = wavelength_cols[0][1]
            actual_wavelength = wavelength_cols[0][2]
            
            # Extract data from the closest wavelength column
            data = df[closest_col].dropna()
            
            # Get corresponding Sample_ID values if column exists
            sample_ids = None
            if 'Sample_ID' in df.columns:
                sample_ids = df['Sample_ID'].iloc[:len(data)].values
            
            # Create dataframe with file info and data
            file_data = pd.DataFrame({
                'file': os.path.basename(csv_file),
                'wavelength': actual_wavelength,
                'sample_id': sample_ids if sample_ids is not None else range(len(data)),
                'value': data.values
            })
            
            all_data.append(file_data)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data found in any CSV files")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Merge with metadata if available
    if metadata_df is not None:
        combined_df = pd.merge(combined_df, metadata_df, on='sample_id', how='left')
    
    return combined_df
