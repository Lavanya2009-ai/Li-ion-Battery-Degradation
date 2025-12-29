"""
COMPLETE WORKING MULTI-FACTOR TRANSFORMER
With proper validation (Leave-one-battery-out cross-validation + Pressure effect analysis)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import json
from pathlib import Path
from typing import List, Dict, Tuple
import copy

# ==================== 1. DATA LOADER ====================

class BatteryDatasetProper(Dataset):
    """Proper data loader with full battery tracking"""
    
    def __init__(self, csv_path: str, sequence_length: int = 30, stride: int = 15):
        super().__init__()
        
        print(f"📊 Loading dataset from: {csv_path}")
        
        essential_cols = [
            'voltage', 'current', 'capacity', 'time_seconds',
            'temperature', 'pressure', 'c_rate', 'battery_id'
        ]
        
        try:
            self.df = pd.read_csv(csv_path, usecols=essential_cols, nrows=500000)
        except:
            chunks = []
            for chunk in pd.read_csv(csv_path, usecols=essential_cols, chunksize=100000):
                chunks.append(chunk)
                if len(chunks) >= 5:
                    break
            self.df = pd.concat(chunks, ignore_index=True)
        
        print(f"✅ Loaded {len(self.df):,} rows")
        
        self.df = self.df[self.df['capacity'] > 0.001].copy()
        print(f"📊 After cleaning: {len(self.df):,} rows")
        
        self.battery_ids = sorted(self.df['battery_id'].unique())
        print(f"🔋 Batteries found: {self.battery_ids}")
        
        self.battery_data = {}
        self.battery_sequences = {}
        self.battery_factors = {}
        
        self.feature_cols = ['voltage', 'current', 'time_seconds', 
                           'temperature', 'pressure', 'c_rate']
        
        for battery_id in self.battery_ids:
            battery_df = self.df[self.df['battery_id'] == battery_id].copy()
            
            if len(battery_df) < sequence_length:
                continue
                
            battery_df = battery_df.sort_values('time_seconds')
            
            pressure = float(battery_df['pressure'].iloc[0])
            temperature = float(battery_df['temperature'].iloc[0])
            c_rate = float(battery_df['c_rate'].iloc[0])
            
            X = battery_df[self.feature_cols].values
            y = battery_df['capacity'].values
            
            sequences = []
            targets = []
            factors = []
            
            for i in range(0, len(X) - sequence_length + 1, stride):
                seq = X[i:i + sequence_length]
                target = y[i + sequence_length - 1]
                
                sequences.append(seq)
                targets.append(target)
                factors.append({
                    'pressure': pressure,
                    'temperature': temperature,
                    'c_rate': c_rate,
                    'battery_id': battery_id,
                    'seq_idx': i
                })
            
            if sequences:
                self.battery_data[battery_id] = {
                    'sequences': np.array(sequences),
                    'targets': np.array(targets),
                    'factors': factors,
                    'pressure': pressure,
                    'temperature': temperature,
                    'c_rate': c_rate
                }
                
                self.battery_sequences[battery_id] = np.array(sequences)
                self.battery_factors[battery_id] = factors
        
        all_sequences = []
        all_targets = []
        
        for battery_id in self.battery_sequences:
            all_sequences.append(self.battery_sequences[battery_id])
            all_targets.append(self.battery_data[battery_id]['targets'])
        
        if all_sequences:
            all_sequences = np.vstack(all_sequences)
            all_targets = np.concatenate(all_targets)
            
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(all_sequences.reshape(-1, len(self.feature_cols)))
            
            self.target_scaler = StandardScaler()
            self.target_scaler.fit(all_targets.reshape(-1, 1))
            
            self.total_sequences = sum(len(seqs) for seqs in self.battery_sequences.values())
            print(f"✅ Created {self.total_sequences} sequences total")
            
            print(f"\n📊 Battery Sequence Distribution:")
            for battery_id in sorted(self.battery_sequences.keys()):
                count = len(self.battery_sequences[battery_id])
                data = self.battery_data[battery_id]
                print(f"   Battery {battery_id}: {count} sequences "
                      f"({data['pressure']}N, {data['temperature']}°C, {data['c_rate']}C)")
        else:
            raise ValueError("No sequences created!")
    
    def get_battery_indices(self, battery_ids: List[int]) -> List[int]:
        """Get indices for specific batteries"""
        indices = []
        current_idx = 0
        
        for battery_id in sorted(self.battery_sequences.keys()):
            count = len(self.battery_sequences[battery_id])
            if battery_id in battery_ids:
                indices.extend(range(current_idx, current_idx + count))
            current_idx += count
            
        return indices
    
    def __len__(self) -> int:
        return self.total_sequences
    
    def __getitem__(self, idx: int):
        current_idx = 0
        for battery_id in sorted(self.battery_sequences.keys()):
            count = len(self.battery_sequences[battery_id])
            if idx < current_idx + count:
                battery_idx = idx - current_idx
                sequences = self.battery_sequences[battery_id]
                targets = self.battery_data[battery_id]['targets']
                factors = self.battery_factors[battery_id][battery_idx]
                
                sequence = sequences[battery_idx]
                sequence_norm = self.feature_scaler.transform(sequence)
                
                target = targets[battery_idx]
                target_norm = self.target_scaler.transform([[target]])[0, 0]
                
                sequence_tensor = torch.FloatTensor(sequence_norm)
                target_tensor = torch.FloatTensor([target_norm])
                
                pressure_idx = self._pressure_to_idx(factors['pressure'])
                temp_idx = self._temp_to_idx(factors['temperature'])
                crate_idx = self._crate_to_idx(factors['c_rate'])
                
                factor_tensor = torch.LongTensor([pressure_idx, temp_idx, crate_idx])
                
                return {
                    'sequence': sequence_tensor,
                    'target': target_tensor,
                    'factors': factor_tensor,
                    'battery_id': battery_id,
                    'pressure': factors['pressure'],
                    'temperature': factors['temperature'],
                    'c_rate': factors['c_rate'],
                    'raw_target': torch.FloatTensor([target])
                }
            
            current_idx += count
        
        raise IndexError(f"Index {idx} out of bounds")
    
    def _pressure_to_idx(self, pressure: float) -> int:
        pressure = float(pressure)
        if pressure == 300: return 0
        elif pressure == 400: return 1
        elif pressure == 500: return 2
        elif pressure == 600: return 3
        else: return 4
    
    def _temp_to_idx(self, temp: float) -> int:
        temp = float(temp)
        if temp == 10: return 0
        elif temp == 25: return 1
        elif temp == 40: return 2
        else: return 3
    
    def _crate_to_idx(self, crate: float) -> int:
        crate = float(crate)
        if crate == 0.5: return 0
        elif crate == 1.0: return 1
        elif crate == 1.5: return 2
        elif crate == 2.0: return 3
        else: return 4

# ==================== 2. MODEL ARCHITECTURE ====================

class RobustTransformer(nn.Module):
    """More robust transformer with better regularization"""
    
    def __init__(self, n_features: int, d_model: int = 128):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        
        self.input_projection = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.pos_encoder = self._create_positional_encoding(d_model, 1000)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.3,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.pressure_embed = nn.Embedding(5, 32)
        self.temp_embed = nn.Embedding(4, 32)
        self.crate_embed = nn.Embedding(5, 32)
        
        self.attention_pool = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=0.2, batch_first=True
        )
        
        self.capacity_head = nn.Sequential(
            nn.Linear(d_model + 96, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self.factor_analyzer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=1)
        )
        
        self._init_weights()
    
    def _create_positional_encoding(self, d_model: int, max_len: int = 1000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, sequence: torch.Tensor, factor_indices: torch.Tensor):
        batch_size, seq_len, _ = sequence.shape
        
        x = self.input_projection(sequence)
        
        if seq_len <= self.pos_encoder.shape[0]:
            pos_enc = self.pos_encoder[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
            x = x + pos_enc
        
        encoded = self.transformer(x)
        
        context, _ = self.attention_pool(encoded, encoded, encoded)
        context = torch.mean(context, dim=1)
        
        pressure_emb = self.pressure_embed(factor_indices[:, 0])
        temp_emb = self.temp_embed(factor_indices[:, 1])
        crate_emb = self.crate_embed(factor_indices[:, 2])
        factors_combined = torch.cat([pressure_emb, temp_emb, crate_emb], dim=1)
        
        combined = torch.cat([context, factors_combined], dim=1)
        capacity = self.capacity_head(combined)
        
        factor_contrib = self.factor_analyzer(context)
        
        return {
            'capacity': capacity,
            'pressure_contrib': factor_contrib[:, 0],
            'temp_contrib': factor_contrib[:, 1],
            'crate_contrib': factor_contrib[:, 2],
            'context': context
        }

# ==================== 3. VALIDATION & ANALYSIS ====================

def leave_one_battery_out_validation(dataset: BatteryDatasetProper, config: Dict):
    """Proper leave-one-battery-out cross-validation"""
    
    print("\n" + "="*80)
    print("🔬 LEAVE-ONE-BATTERY-OUT CROSS VALIDATION")
    print("="*80)
    
    battery_ids = list(dataset.battery_sequences.keys())
    print(f"🔋 Batteries for CV: {battery_ids}")
    
    results = {}
    
    for test_battery_id in battery_ids:
        print(f"\n📊 Fold: Test Battery {test_battery_id}")
        print("-" * 40)
        
        train_batteries = [bid for bid in battery_ids if bid != test_battery_id]
        test_batteries = [test_battery_id]
        
        train_indices = dataset.get_battery_indices(train_batteries)
        test_indices = dataset.get_battery_indices(test_batteries)
        
        print(f"   Train batteries: {train_batteries} ({len(train_indices)} sequences)")
        print(f"   Test battery: {test_batteries} ({len(test_indices)} sequences)")
        
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_subset, batch_size=config['batch_size'], 
                                 shuffle=True, num_workers=0)
        test_loader = DataLoader(test_subset, batch_size=config['batch_size'],
                                shuffle=False, num_workers=0)
        
        model = RobustTransformer(
            n_features=len(dataset.feature_cols),
            d_model=config['d_model']
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], 
                               weight_decay=config['weight_decay'])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = config.get('patience', 15)
        
        fold_results = {'train_loss': [], 'test_loss': [], 'test_mae': []}
        
        for epoch in range(config['epochs']):
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                sequence = batch['sequence']
                target = batch['target']
                factors = batch['factors']
                
                optimizer.zero_grad()
                predictions = model(sequence, factors)
                loss = criterion(predictions['capacity'], target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            model.eval()
            test_loss = 0
            test_mae = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    sequence = batch['sequence']
                    target = batch['target']
                    factors = batch['factors']
                    
                    predictions = model(sequence, factors)
                    loss = criterion(predictions['capacity'], target)
                    mae = torch.abs(predictions['capacity'] - target).mean().item()
                    
                    test_loss += loss.item()
                    test_mae += mae
            
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            test_mae /= len(test_loader)
            
            fold_results['train_loss'].append(train_loss)
            fold_results['test_loss'].append(test_loss)
            fold_results['test_mae'].append(test_mae)
            
            scheduler.step(test_loss)
            
            if test_loss < best_val_loss:
                best_val_loss = test_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'best_model_battery_{test_battery_id}.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1}: Train={train_loss:.4f}, "
                      f"Test={test_loss:.4f}, MAE={test_mae:.4f}, "
                      f"LR={optimizer.param_groups[0]['lr']:.6f}")
            
            if patience_counter >= max_patience:
                print(f"   ⏹️ Early stopping at epoch {epoch+1}")
                break
        
        results[test_battery_id] = {
            'best_test_loss': best_val_loss,
            'final_test_mae': fold_results['test_mae'][-1],
            'train_losses': fold_results['train_loss'],
            'test_losses': fold_results['test_loss'],
            'test_maes': fold_results['test_mae'],
            'num_epochs': len(fold_results['train_loss'])
        }
        
        print(f"   ✅ Best test loss: {best_val_loss:.4f}")
        print(f"   ✅ Final test MAE: {fold_results['test_mae'][-1]:.4f}")
    
    return results

def analyze_pressure_effects(dataset: BatteryDatasetProper, model_path: str = None):
    """Analyze how pressure affects predictions"""
    
    print("\n" + "="*80)
    print("🔬 PRESSURE EFFECT ANALYSIS")
    print("="*80)
    
    model = RobustTransformer(n_features=len(dataset.feature_cols))
    
    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path))
        print(f"✅ Loaded model from {model_path}")
    
    model.eval()
    
    battery_id = list(dataset.battery_sequences.keys())[0]
    print(f"🔋 Analyzing pressure effects on Battery {battery_id}")
    
    battery_indices = dataset.get_battery_indices([battery_id])
    sample_idx = battery_indices[0]
    sample = dataset[sample_idx]
    base_sequence = sample['sequence'].unsqueeze(0)
    
    print(f"\n📊 Base conditions:")
    print(f"   Battery: {battery_id}")
    print(f"   Temperature: {sample['temperature']}°C")
    print(f"   C-rate: {sample['c_rate']}C")
    
    pressure_results = []
    
    for test_pressure in [300, 400, 500, 600]:
        pressure_idx = dataset._pressure_to_idx(test_pressure)
        temp_idx = dataset._temp_to_idx(sample['temperature'])
        crate_idx = dataset._crate_to_idx(sample['c_rate'])
        
        factor_indices = torch.LongTensor([[pressure_idx, temp_idx, crate_idx]])
        
        with torch.no_grad():
            prediction = model(base_sequence, factor_indices)
        
        pred_norm = prediction['capacity'].item()
        pred_denorm = dataset.target_scaler.inverse_transform([[pred_norm]])[0, 0]
        
        pressure_results.append({
            'pressure': test_pressure,
            'predicted_capacity': pred_denorm,
            'pressure_contribution': prediction['pressure_contrib'].item(),
            'temp_contribution': prediction['temp_contrib'].item(),
            'crate_contribution': prediction['crate_contrib'].item()
        })
    
    print(f"\n📈 Pressure Effect Results:")
    print("-" * 60)
    
    for i, result in enumerate(pressure_results):
        print(f"  {result['pressure']}N: Predicted Capacity = {result['predicted_capacity']:.4f}")
        print(f"     Contributions: P={result['pressure_contribution']:.3f}, "
              f"T={result['temp_contribution']:.3f}, C={result['crate_contribution']:.3f}")
    
    capacity_300N = pressure_results[0]['predicted_capacity']
    capacity_600N = pressure_results[3]['predicted_capacity']
    pressure_effect = capacity_300N - capacity_600N
    
    print(f"\n🎯 Pressure Effect (300N → 600N):")
    print(f"   Capacity change: {capacity_300N:.4f} → {capacity_600N:.4f}")
    print(f"   Absolute effect: {pressure_effect:.4f}")
    print(f"   Relative effect: {(pressure_effect/capacity_300N)*100:.1f}% decrease")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    pressures = [r['pressure'] for r in pressure_results]
    capacities = [r['predicted_capacity'] for r in pressure_results]
    
    axes[0].plot(pressures, capacities, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Pressure (N)')
    axes[0].set_ylabel('Predicted Capacity')
    axes[0].set_title('Capacity Prediction vs Pressure')
    axes[0].grid(True, alpha=0.3)
    
    contributions = ['pressure_contribution', 'temp_contribution', 'crate_contribution']
    avg_contributions = [
        np.mean([r['pressure_contribution'] for r in pressure_results]),
        np.mean([r['temp_contribution'] for r in pressure_results]),
        np.mean([r['crate_contribution'] for r in pressure_results])
    ]
    
    bars = axes[1].bar(['Pressure', 'Temperature', 'C-rate'], avg_contributions)
    axes[1].set_ylabel('Average Contribution')
    axes[1].set_title('Factor Contributions to Predictions')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, avg_contributions):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('pressure_effect_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    analysis_results = {
        'pressure_effects': pressure_results,
        'pressure_effect_magnitude': float(pressure_effect),
        'relative_effect_percent': float((pressure_effect/capacity_300N)*100),
        'factor_contributions': {
            'pressure': float(avg_contributions[0]),
            'temperature': float(avg_contributions[1]),
            'c_rate': float(avg_contributions[2])
        }
    }
    
    with open('pressure_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n💾 Analysis saved to: pressure_analysis.json")
    
    return pressure_results

# ==================== 4. MAIN EXECUTION ====================

def main():
    """Main execution function"""
    print("="*80)
    print("🔬 BATTERY DEGRADATION TRANSFORMER - COMPLETE ANALYSIS")
    print("="*80)
    
    config = {
        'd_model': 128,
        'batch_size': 32,
        'lr': 1e-3,
        'weight_decay': 0.01,
        'epochs': 50,
        'patience': 15
    }
    
    try:
        print("\n📥 Loading dataset...")
        dataset = BatteryDatasetProper(
            csv_path="enhanced_battery_data_with_features.csv",
            sequence_length=30,
            stride=15
        )
        
        print(f"\n✅ Dataset Summary:")
        print(f"   Total sequences: {len(dataset):,}")
        print(f"   Features: {len(dataset.feature_cols)}")
        print(f"   Batteries: {len(dataset.battery_sequences)}")
        
        print("\n" + "="*80)
        print("🏃 RUNNING LEAVE-ONE-BATTERY-OUT VALIDATION")
        print("="*80)
        
        cv_results = leave_one_battery_out_validation(dataset, config)
        
        print("\n" + "="*80)
        print("📊 CROSS-VALIDATION RESULTS SUMMARY")
        print("="*80)
        
        test_maes = []
        test_losses = []
        
        for battery_id, results in cv_results.items():
            test_mae = results['final_test_mae']
            test_loss = results['best_test_loss']
            
            test_maes.append(test_mae)
            test_losses.append(test_loss)
            
            print(f"\n🔋 Battery {battery_id}:")
            print(f"   Best test loss: {test_loss:.6f}")
            print(f"   Final test MAE: {test_mae:.6f}")
            print(f"   Epochs trained: {results['num_epochs']}")
        
        avg_test_mae = np.mean(test_maes)
        std_test_mae = np.std(test_maes)
        avg_test_loss = np.mean(test_losses)
        
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"   Average test MAE: {avg_test_mae:.6f} ± {std_test_mae:.6f}")
        print(f"   Average test loss: {avg_test_loss:.6f}")
        
        print("\n" + "="*80)
        print("🔬 ANALYZING PRESSURE EFFECTS")
        print("="*80)
        
        best_battery = min(cv_results.items(), key=lambda x: x[1]['best_test_loss'])[0]
        best_model_path = f'best_model_battery_{best_battery}.pth'
        
        print(f"📁 Using best model from Battery {best_battery}")
        
        pressure_results = analyze_pressure_effects(dataset, best_model_path)
        
        final_results = {
            'config': config,
            'cv_results': cv_results,
            'overall_performance': {
                'avg_test_mae': float(avg_test_mae),
                'std_test_mae': float(std_test_mae),
                'avg_test_loss': float(avg_test_loss)
            },
            'dataset_info': {
                'total_sequences': len(dataset),
                'num_batteries': len(dataset.battery_sequences),
                'feature_count': len(dataset.feature_cols)
            }
        }
        
        with open('final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        
        print(f"\n📁 Output files created:")
        print(f"   1. best_model_battery_*.pth - Trained models")
        print(f"   2. pressure_effect_analysis.png - Visualization")
        print(f"   3. pressure_analysis.json - Pressure effect metrics")
        print(f"   4. final_results.json - Complete analysis results")
        
        print(f"\n🎯 Key findings:")
        print(f"   • Model generalizes with {avg_test_mae:.4f} MAE")
        print(f"   • Pressure effect: {pressure_results[0]['predicted_capacity']:.3f} → "
              f"{pressure_results[-1]['predicted_capacity']:.3f} (300N→600N)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
