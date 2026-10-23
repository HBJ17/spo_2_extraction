import pandas as pd
import os

def convert_excel_to_ground_truth(excel_path, output_csv_path):
    print(f"Reading {excel_path}...")
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        return
        
    if 'Digital_ID' not in df.columns or 'SPO2' not in df.columns:
        print("Expected columns 'Digital_ID' and 'SPO2' not found in the Excel sheet.")
        return
        
    # Select only the relevant columns and drop rows with missing SpO2 values
    gt_df = df[['Digital_ID', 'SPO2']].dropna()
    
    # Rename to match our pipeline's expectations (even though it's flexible, being explicit is good)
    gt_df = gt_df.rename(columns={'Digital_ID': 'filename', 'SPO2': 'spo2'})
    
    # Save as CSV
    gt_df.to_csv(output_csv_path, index=False)
    print(f"Successfully converted! Saved {len(gt_df)} valid ground truth entries to {output_csv_path}.")

if __name__ == "__main__":
    excel_file = r"D:\Studies\ML\DATA\Video\SSN_MIKENKO_DATASET.xlsx"
    output_csv = r"d:\Studies\ML\spo2_extractor\Spo2_ML\ground_truth.csv"
    convert_excel_to_ground_truth(excel_file, output_csv)
