"""
BATTERY DATASET LOADER
Combines all battery files (.xls) into a combined dataset for model building
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

class BatteryDatasetLoader:
    """Loader for hierarchical battery dataset"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.files = self._find_all_excel_files()
        self._print_directory_structure()
        print(f"\n Found {len(self.files)} Excel files")
    
    def _print_directory_structure(self, max_depth: int = 4):
        """Print the hierarchical directory structure"""
        print("\n" + "="*70)
        print("DIRECTORY STRUCTURE VISUALIZATION")
        print("="*70)
        
        def print_tree(path: Path, prefix: str = "", depth: int = 0, is_last: bool = True):
            if depth > max_depth:
                return
            
            marker = "└── " if is_last else "├── "
            
            if depth == 0:
                print(f"{path.name}/")
            else:
                print(f"{prefix}{marker}{path.name}/" if path.is_dir() else f"{prefix}{marker}{path.name}")
            
            if path.is_dir():
                try:
                    items = sorted(list(path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
                except:
                    return
                
                new_prefix = prefix + ("    " if is_last else "│   ")
                
                for i, item in enumerate(items):
                    is_last_item = (i == len(items) - 1)
                    print_tree(item, new_prefix, depth + 1, is_last_item)
        
        print_tree(self.base_path)
    
    def _find_all_excel_files(self) -> List[Path]:
        """Find all .xls files in the hierarchical structure"""
        excel_files = []
        
        print(f"\n Scanning directory: {self.base_path}")
        
        # First level: Temperature directories
        temp_dirs = sorted(list(self.base_path.glob("*d")))
        print(f"   Found {len(temp_dirs)} temperature directories: {[d.name for d in temp_dirs]}")
        
        for temp_dir in temp_dirs:
            if temp_dir.is_dir():
                # Second level: Pressure directories
                pressure_dirs = sorted(list(temp_dir.glob("*N")))
                print(f"   ├── {temp_dir.name}: {len(pressure_dirs)} pressure directories")
                
                for pressure_dir in pressure_dirs:
                    if pressure_dir.is_dir():
                        # Third level: C-rate directories
                        crate_dirs = sorted(list(pressure_dir.glob("*C")))
                        print(f"   │   ├── {pressure_dir.name}: {len(crate_dirs)} C-rate directories")
                        
                        for crate_dir in crate_dirs:
                            if crate_dir.is_dir():
                                # Fourth level: Excel files
                                battery_files = list(crate_dir.glob("*.xls")) + list(crate_dir.glob("*.xlsx"))
                                excel_files.extend(battery_files)
                                print(f"   │   │   ├── {crate_dir.name}: {len(battery_files)} Excel files")
        
        print(f"\n Total Excel files found: {len(excel_files)}")
        return sorted(excel_files)
    
    def parse_conditions_from_path(self, file_path: Path) -> Dict:
        """Extract experimental conditions from file path"""
        parts = file_path.parts
        conditions = {}
        
        for part in parts:
            part = str(part).lower()
            
            # Temperature (e.g., '10d')
            if part.endswith('d') and part[:-1].replace('.', '').isdigit():
                try:
                    conditions['temperature'] = float(part[:-1])
                except:
                    conditions['temperature'] = 25.0
            
            # Pressure (e.g., '300n')
            elif part.endswith('n') and part[:-1].replace('.', '').isdigit():
                try:
                    conditions['pressure'] = int(part[:-1])
                except:
                    conditions['pressure'] = 300
            
            # C-rate (e.g., '0.5c')
            elif part.endswith('c'):
                c_val = part[:-1]
                if c_val == '0.5':
                    conditions['c_rate'] = 0.5
                elif c_val.replace('.', '').isdigit():
                    conditions['c_rate'] = float(c_val)
        
        # Battery ID from filename
        filename = file_path.stem
        match = re.search(r'No\.\s*(\d+)', filename, re.IGNORECASE)
        if match:
            conditions['battery_id'] = int(match.group(1))
        else:
            match = re.search(r'(\d+)$', filename)
            conditions['battery_id'] = int(match.group(1)) if match else 1
        
        return conditions
    
    def load_single_file(self, file_path: Path) -> Tuple[pd.DataFrame, Dict]:
        """Load a single Excel file with its conditions"""
        try:
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Parse conditions from path
            conditions = self.parse_conditions_from_path(file_path)
            
            # Clean column names
            df.columns = [str(col).strip() for col in df.columns]
            
            # Standardize column names
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
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Convert units
            df['current'] = df['current_mA'] / 1000.0  # mA to A
            df['capacity'] = df['capacity_mAh'] / 1000.0  # mAh to Ah
            df['power'] = df['power_mW'] / 1000.0  # mW to W
            
            # Parse time string to seconds
            df['time_seconds'] = df['time_str'].apply(self._parse_time_to_seconds)
            df['time_normalized'] = df['time_seconds'] - df['time_seconds'].iloc[0]
            
            # Add experimental conditions
            df['temperature'] = conditions['temperature']
            df['pressure'] = conditions['pressure']
            df['c_rate'] = conditions['c_rate']
            df['battery_id'] = conditions['battery_id']
            df['file_name'] = file_path.name
            
            # Identify cycles
            df['cycle'] = self._identify_cycles(df)
            
            print(f" Loaded: {file_path.name}")
            print(f"   Rows: {len(df):,}")
            print(f"   Conditions: {conditions['temperature']}°C, {conditions['pressure']}N, "
                  f"{conditions['c_rate']}C, Battery {conditions['battery_id']}")
            
            return df, conditions
            
        except Exception as e:
            print(f" Error loading {file_path.name}: {e}")
            return None, {}
    
    def _parse_time_to_seconds(self, time_str):
        """Convert 'h:min:s.ms' to seconds"""
        try:
            if pd.isna(time_str):
                return 0.0
            
            time_str = str(time_str)
            
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 3:
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    
                    seconds_part = parts[2]
                    if '.' in seconds_part:
                        seconds, milliseconds = map(float, seconds_part.split('.'))
                        milliseconds = milliseconds / (10 ** len(str(int(milliseconds))))
                    else:
                        seconds = float(seconds_part)
                        milliseconds = 0.0
                    
                    return hours * 3600 + minutes * 60 + seconds + milliseconds
            
            return float(time_str)
        except:
            return 0.0
    
    def _identify_cycles(self, df):
        """Identify charge/discharge cycles"""
        cycles = []
        cycle_num = 0
        
        if 'current' not in df.columns:
            return [0] * len(df)
        
        for i in range(len(df)):
            if i == 0:
                cycles.append(cycle_num)
            else:
                current_i = df['current'].iloc[i]
                current_prev = df['current'].iloc[i-1]
                
                if current_i * current_prev < 0 and abs(current_i) > 0.001:
                    cycle_num += 1
                cycles.append(cycle_num)
        
        return cycles
    
    def analyze_dataset(self):
        """Analyze the complete dataset"""
        print("\n" + "="*70)
        print("DATASET ANALYSIS")
        print("="*70)
        
        condition_counts = defaultdict(int)
        temperature_counts = defaultdict(int)
        pressure_counts = defaultdict(int)
        crate_counts = defaultdict(int)
        total_rows = 0
        
        print("\n Scanning all files...")
        for i, file in enumerate(self.files[:10]):
            conditions = self.parse_conditions_from_path(file)
            
            key = f"{conditions['temperature']}°C_{conditions['pressure']}N_{conditions['c_rate']}C"
            condition_counts[key] += 1
            
            temperature_counts[conditions['temperature']] += 1
            pressure_counts[conditions['pressure']] += 1
            crate_counts[conditions['c_rate']] += 1
            
            df, _ = self.load_single_file(file)
            if df is not None:
                total_rows += len(df)
        
        print(f"\n EXPERIMENTAL DESIGN:")
        print(f"   • Temperatures: {sorted(temperature_counts.keys())}")
        print(f"   • Pressures: {sorted(pressure_counts.keys())}")
        print(f"   • C-rates: {sorted(crate_counts.keys())}")
        
        avg_rows_per_file = total_rows // 10 if total_rows > 0 else 0
        estimated_total_rows = avg_rows_per_file * len(self.files)
        
        print(f"\n STATISTICAL ESTIMATES:")
        print(f"   Average rows per file: {avg_rows_per_file:,}")
        print(f"   Estimated total dataset size: {estimated_total_rows:,} rows")
        
        return {
            'total_files': len(self.files),
            'estimated_total_rows': estimated_total_rows,
            'temperature_counts': dict(temperature_counts),
            'pressure_counts': dict(pressure_counts),
            'crate_counts': dict(crate_counts)
        }
    
    def load_all_files(self, max_files: Optional[int] = None) -> Tuple[List[pd.DataFrame], List[Dict]]:
        """Load all files and return combined data"""
        print("\n" + "="*70)
        print(f"LOADING ALL FILES{' (FIRST ' + str(max_files) + ')' if max_files else ''}")
        print("="*70)
        
        all_dataframes = []
        all_conditions = []
        total_rows = 0
        failed_files = 0
        
        files_to_load = self.files[:max_files] if max_files else self.files
        
        for i, file in enumerate(files_to_load):
            print(f"\n[{i+1}/{len(files_to_load)}] ", end="")
            df, conditions = self.load_single_file(file)
            
            if df is not None:
                all_dataframes.append(df)
                all_conditions.append(conditions)
                total_rows += len(df)
            else:
                failed_files += 1
        
        print(f"\n" + "="*70)
        print("LOADING SUMMARY")
        print("="*70)
        print(f" Successfully loaded: {len(all_dataframes)} files")
        print(f" Failed to load: {failed_files} files")
        print(f" Total rows loaded: {total_rows:,}")
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            print(f"📈 Combined dataset shape: {combined_df.shape}")
            print(f"📋 Combined columns: {len(combined_df.columns)}")
            
            return combined_df, all_conditions
        else:
            print(" No data loaded!")
            return None, []

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BATTERY DATASET LOADER - COMPLETE HIERARCHICAL STRUCTURE")
    print("="*70)
    
    data_path = "C:/Users/SASTRA/Videos/libd"
    print(f"\n Data path: {data_path}")
    
    loader = BatteryDatasetLoader(data_path)
    analysis = loader.analyze_dataset()
    
    if loader.files:
        print("\n" + "="*70)
        print("SAMPLE FILE DETAILED INSPECTION")
        print("="*70)
        
        sample_df, conditions = loader.load_single_file(loader.files[0])
        
        if sample_df is not None:
            print(f"\n📄 SAMPLE DATAFRAME INFO:")
            print(f"   Shape: {sample_df.shape} (rows × columns)")
            
            important_cols = ['time_str', 'voltage', 'current', 'capacity', 
                            'temperature', 'pressure', 'c_rate', 'battery_id', 'cycle']
            available_cols = [c for c in important_cols if c in sample_df.columns]
            print(sample_df[available_cols].head(3).to_string())
            
            response = input("\nLoad ALL files to create combined dataset? (y/n): ")
            
            if response.lower() == 'y':
                max_files = min(50, len(loader.files))
                combined_df, all_conditions = loader.load_all_files(max_files=max_files)
                
                if combined_df is not None:
                    output_path = "combined_battery_data.csv"
                    combined_df.to_csv(output_path, index=False)
                    print(f"\n Combined data saved to: {output_path}")
                    
                    conditions_df = pd.DataFrame(all_conditions)
                    conditions_df.to_csv("battery_conditions.csv", index=False)
                    print(f" Conditions metadata saved to: battery_conditions.csv")
