"""
COMPLETE FEATURE ENGINEERING FOR ALL BATTERY FILES
Calculates features for each file and adds them to the combined dataset
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import time
from tqdm import tqdm
import gc

class BatteryFeatureEngineer:
    """Engineer features for all battery files"""
    
    def __init__(self):
        self.feature_names = []
    
    def process_all_files(self, data_root="C:/Users/SASTRA/Videos/libd"):
        """Process all Excel files and create enhanced dataset"""
        print(f"\n🔍 SCANNING FOR BATTERY FILES in: {data_root}")
        
        excel_files = []
        for ext in ['.xls', '.xlsx']:
            excel_files.extend(list(Path(data_root).rglob(f"*{ext}")))
        
        print(f"✅ Found {len(excel_files)} Excel files")
        
        if len(excel_files) == 0:
            print("❌ No Excel files found!")
            return None
        
        all_data = []
        failed_files = []
        
        print(f"\n📊 PROCESSING {len(excel_files)} FILES...")
        
        for i, file_path in enumerate(tqdm(excel_files, desc="Processing files")):
            try:
                file_data = self.process_single_file(file_path)
                
                if file_data is not None and not file_data.empty:
                    file_data_with_features = self.calculate_file_features(file_data, str(file_path))
                    all_data.append(file_data_with_features)
                    
                    if (i + 1) % 10 == 0:
                        print(f"   Processed {i+1}/{len(excel_files)} files...")
                        gc.collect()
                else:
                    failed_files.append(str(file_path))
                    
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
                failed_files.append(str(file_path))
        
        if all_data:
            print(f"\n✅ SUCCESSFULLY PROCESSED {len(all_data)} files")
            if failed_files:
                print(f"⚠️  Failed to process {len(failed_files)} files")
            
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n📈 COMBINED DATASET:")
            print(f"   Total rows: {len(combined_df):,}")
            print(f"   Total columns: {len(combined_df.columns)}")
            
            return combined_df
        else:
            print("❌ No data processed!")
            return None
    
    def process_single_file(self, file_path):
        """Process a single battery Excel file"""
        try:
            df = pd.read_excel(file_path)
            file_info = self.parse_file_info(file_path)
            
            df = self.standardize_columns(df)
            
            for key, value in file_info.items():
                df[key] = value
            
            df['source_file'] = str(file_path.name)
            df['full_path'] = str(file_path)
            
            return df
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    def parse_file_info(self, file_path):
        """Parse experimental conditions from file path"""
        parts = str(file_path).split('\\')
        
        info = {
            'temperature': 25.0,
            'pressure': 300,
            'c_rate': 1.0,
            'battery_id': 1
        }
        
        for part in parts:
            part_lower = part.lower()
            
            if 'd' in part_lower and part_lower.replace('d', '').isdigit():
                info['temperature'] = float(part_lower.replace('d', ''))
            elif 'n' in part_lower and part_lower.replace('n', '').isdigit():
                info['pressure'] = int(part_lower.replace('n', ''))
            elif 'c' in part_lower:
                c_str = part_lower.replace('c', '')
                if c_str == '0.5':
                    info['c_rate'] = 0.5
                elif c_str.replace('.', '').isdigit():
                    info['c_rate'] = float(c_str)
            elif 'battery' in part_lower and 'no' in part_lower:
                import re
                match = re.search(r'(\d+)', part_lower)
                if match:
                    info['battery_id'] = int(match.group(1))
        
        return info
    
    def standardize_columns(self, df):
        """Standardize column names and units"""
        df_clean = df.copy()
        
        column_mapping = {
            'Step Name': 'step',
            'Record Number': 'record_num',
            'Record Time(h:min:s.ms)': 'time_str',
            'Voltage(V)': 'voltage',
            'Current(mA)': 'current_mA',
            'Capacity(mAh)': 'capacity_mAh',
            'Power(mWh)': 'power_mW',
            'Absolute Time': 'abs_time'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df_clean.columns:
                df_clean = df_clean.rename(columns={old_col: new_col})
        
        if 'current_mA' in df_clean.columns:
            df_clean['current'] = df_clean['current_mA'] / 1000.0
        
        if 'capacity_mAh' in df_clean.columns:
            df_clean['capacity'] = df_clean['capacity_mAh'] / 1000.0
        
        if 'power_mW' in df_clean.columns:
            df_clean['power'] = df_clean['power_mW'] / 1000.0
        
        if 'time_str' in df_clean.columns:
            df_clean['time_seconds'] = df_clean['time_str'].apply(self.time_to_seconds)
            df_clean['time_normalized'] = df_clean['time_seconds'] - df_clean['time_seconds'].iloc[0]
        
        return df_clean
    
    def time_to_seconds(self, time_str):
        """Convert time string to seconds"""
        try:
            if pd.isna(time_str):
                return 0.0
            
            parts = str(time_str).split(':')
            if len(parts) == 3:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            else:
                return float(time_str)
        except:
            return 0.0
    
    def calculate_file_features(self, df, file_path):
        """Calculate comprehensive features for a single file"""
        df_with_features = df.copy()
        
        required_cols = ['voltage', 'current', 'capacity', 'time_seconds']
        for col in required_cols:
            if col not in df_with_features.columns:
                print(f"⚠️  Missing {col} in {file_path}")
                return df
        
        voltage = df_with_features['voltage'].values
        current = df_with_features['current'].values
        
        # 1. VOLTAGE FEATURES
        window_size = min(100, len(voltage))
        if window_size > 10:
            df_with_features['voltage_rolling_mean'] = df_with_features['voltage'].rolling(window_size, center=True).mean()
            df_with_features['voltage_rolling_std'] = df_with_features['voltage'].rolling(window_size, center=True).std()
            df_with_features['voltage_gradient'] = np.gradient(voltage)
        
        # 2. CURRENT FEATURES
        df_with_features['is_charging'] = (current > 0.001).astype(int)
        df_with_features['is_discharging'] = (current < -0.001).astype(int)
        df_with_features['is_resting'] = ((current >= -0.001) & (current <= 0.001)).astype(int)
        df_with_features['current_abs'] = np.abs(current)
        
        # 3. CAPACITY FEATURES
        if 'capacity' in df_with_features.columns:
            capacity = df_with_features['capacity'].values
            
            if len(capacity) > 1:
                capacity_diff = np.diff(capacity, prepend=capacity[0])
                df_with_features['capacity_change'] = capacity_diff
                df_with_features['cumulative_capacity'] = np.cumsum(np.abs(capacity_diff))
        
        # 4. TIME-BASED FEATURES
        if 'time_seconds' in df_with_features.columns:
            time_arr = df_with_features['time_seconds'].values
            
            if len(time_arr) > 1:
                time_diff = np.diff(time_arr, prepend=time_arr[0])
                df_with_features['time_interval'] = time_diff
                df_with_features['elapsed_time'] = time_arr - time_arr[0]
        
        # 5. POWER AND ENERGY FEATURES
        if 'voltage' in df_with_features.columns and 'current' in df_with_features.columns:
            df_with_features['instant_power'] = df_with_features['voltage'] * df_with_features['current']
            
            if 'time_interval' in df_with_features.columns:
                df_with_features['energy_increment'] = df_with_features['instant_power'] * df_with_features['time_interval']
                df_with_features['cumulative_energy'] = np.cumsum(df_with_features['energy_increment'].fillna(0))
        
        # 6. STATE INDICATORS
        states = []
        current_state = 0
        consecutive_counts = 0
        
        for i in range(len(current)):
            if current[i] > 0.01:
                if current_state != 1:
                    current_state = 1
                    consecutive_counts = 1
                else:
                    consecutive_counts += 1
            elif current[i] < -0.01:
                if current_state != 2:
                    current_state = 2
                    consecutive_counts = 1
                else:
                    consecutive_counts += 1
            else:
                if current_state != 3:
                    current_state = 3
                    consecutive_counts = 1
                else:
                    consecutive_counts += 1
            
            if consecutive_counts >= 5:
                states.append(current_state)
            else:
                states.append(0)
        
        df_with_features['battery_state'] = states[:len(df_with_features)]
        
        # Fill NaN values
        df_with_features = df_with_features.fillna(method='ffill').fillna(method='bfill')
        
        return df_with_features
    
    def save_enhanced_dataset(self, df, output_file="enhanced_battery_data.csv"):
        """Save the enhanced dataset with all features"""
        print(f"\n💾 SAVING ENHANCED DATASET...")
        
        if df is None or df.empty:
            print("❌ No data to save!")
            return
        
        df.to_csv(output_file, index=False)
        
        print(f"✅ Enhanced dataset saved to: {output_file}")
        print(f"   Total rows: {len(df):,}")
        print(f"   Total columns: {len(df.columns)}")
        
        return output_file

def main():
    """Main execution function"""
    print("="*80)
    print("ENHANCED BATTERY DATASET CREATION")
    print("="*80)
    
    engineer = BatteryFeatureEngineer()
    
    print("\n🚀 STEP 1: PROCESSING ALL 288 FILES")
    data_root = "C:/Users/SASTRA/Videos/libd"
    
    combined_df = engineer.process_all_files(data_root)
    
    if combined_df is None:
        print("❌ Failed to process files!")
        return
    
    print("\n🚀 STEP 2: CREATING ENHANCED DATASET")
    output_file = "enhanced_battery_data_with_features.csv"
    engineer.save_enhanced_dataset(combined_df, output_file)
    
    print("\n" + "="*80)
    print("✅ ENHANCED DATASET CREATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n⏱️  Total execution time: {end_time - start_time:.1f} seconds")
