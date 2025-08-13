# scripts/test_features.py - Enhanced Version
"""
Enhanced feature testing script with comprehensive analysis and validation.

Features:
- Performance benchmarking
- Memory usage monitoring
- Data quality checks
- Feature distribution analysis
- Correlation analysis
- Missing data assessment
"""

import pandas as pd
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from fetch_data import fetch_matches
except ImportError:
    print("Warning: fetch_data module not found. Please ensure fetch_data.py exists.")
    
from features import build_feature_matrix, get_feature_importance_groups


def get_memory_usage():
    """Get current memory usage in MB."""
    return psutil.virtual_memory().used / (1024 * 1024)


def analyze_data_quality(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
    """Comprehensive data quality analysis."""
    print(f"\n=== {name} Quality Analysis ===")
    
    analysis = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_data': {},
        'duplicates': df.duplicated().sum(),
        'unique_values': {}
    }
    
    # Missing data analysis
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    analysis['missing_data'] = dict(zip(missing.index, zip(missing.values, missing_pct.values)))
    
    # Remove columns with no missing data from display
    missing_cols = {k: v for k, v in analysis['missing_data'].items() if v[0] > 0}
    
    print(f"Shape: {analysis['shape']}")
    print(f"Memory usage: {analysis['memory_usage_mb']:.2f} MB")
    print(f"Data types: {analysis['dtypes']}")
    print(f"Duplicates: {analysis['duplicates']}")
    
    if missing_cols:
        print("\nMissing data:")
        for col, (count, pct) in missing_cols.items():
            print(f"  {col}: {count} ({pct}%)")
    else:
        print("\nNo missing data found!")
    
    # Unique values for categorical-like columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_count = df[col].nunique()
        analysis['unique_values'][col] = unique_count
        print(f"Unique values in {col}: {unique_count}")
    
    return analysis


def analyze_features(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Analyze feature characteristics and relationships."""
    print("\n=== Feature Analysis ===")
    
    analysis = {}
    
    # Basic statistics
    analysis['feature_count'] = len(X.columns)
    analysis['numeric_features'] = len(X.select_dtypes(include=[np.number]).columns)
    analysis['categorical_features'] = len(X.select_dtypes(exclude=[np.number]).columns)
    
    print(f"Total features: {analysis['feature_count']}")
    print(f"Numeric features: {analysis['numeric_features']}")
    print(f"Categorical features: {analysis['categorical_features']}")
    
    # Feature statistics
    numeric_features = X.select_dtypes(include=[np.number])
    
    if len(numeric_features.columns) > 0:
        analysis['feature_stats'] = {
            'mean': numeric_features.mean().to_dict(),
            'std': numeric_features.std().to_dict(),
            'min': numeric_features.min().to_dict(),
            'max': numeric_features.max().to_dict(),
            'skewness': numeric_features.skew().to_dict(),
            'kurtosis': numeric_features.kurtosis().to_dict()
        }
        
        # Identify potentially problematic features
        problematic_features = []
        
        # Features with zero variance
        zero_var_features = numeric_features.columns[numeric_features.var() == 0].tolist()
        if zero_var_features:
            problematic_features.extend(zero_var_features)
            print(f"\nZero variance features: {zero_var_features}")
        
        # Features with extreme skewness (|skew| > 3)
        high_skew_features = numeric_features.columns[abs(numeric_features.skew()) > 3].tolist()
        if high_skew_features:
            print(f"Highly skewed features: {high_skew_features}")
        
        # Features with extreme kurtosis (|kurtosis| > 10)
        high_kurt_features = numeric_features.columns[abs(numeric_features.kurtosis()) > 10].tolist()
        if high_kurt_features:
            print(f"High kurtosis features: {high_kurt_features}")
        
        analysis['problematic_features'] = {
            'zero_variance': zero_var_features,
            'high_skewness': high_skew_features,
            'high_kurtosis': high_kurt_features
        }
    
    # Target distribution
    if y is not None:
        target_dist = y.value_counts().sort_index()
        target_pct = (target_dist / len(y) * 100).round(2)
        analysis['target_distribution'] = {
            'counts': target_dist.to_dict(),
            'percentages': target_pct.to_dict()
        }
        
        print(f"\nTarget distribution:")
        class_names = ['Home Win', 'Draw', 'Away Win']
        for i, (count, pct) in enumerate(zip(target_dist.values, target_pct.values)):
            print(f"  {class_names[i]}: {count} ({pct}%)")
    
    return analysis


def analyze_correlations(X: pd.DataFrame, y: pd.Series, top_n: int = 20) -> Dict[str, Any]:
    """Analyze feature correlations and relationships with target."""
    print(f"\n=== Correlation Analysis (Top {top_n}) ===")
    
    numeric_features = X.select_dtypes(include=[np.number])
    
    if len(numeric_features.columns) == 0:
        print("No numeric features found for correlation analysis.")
        return {}
    
    # Feature-target correlations
    target_corrs = {}
    for col in numeric_features.columns:
        try:
            corr = np.corrcoef(numeric_features[col].fillna(0), y)[0, 1]
            if not np.isnan(corr):
                target_corrs[col] = abs(corr)
        except:
            continue
    
    # Sort by absolute correlation with target
    sorted_corrs = sorted(target_corrs.items(), key=lambda x: x[1], reverse=True)
    
    print("Top features by correlation with target:")
    for i, (feature, corr) in enumerate(sorted_corrs[:top_n], 1):
        print(f"  {i:2d}. {feature}: {corr:.4f}")
    
    # Feature-feature correlations (identify multicollinearity)
    corr_matrix = numeric_features.corr().abs()
    
    # Find high correlations (>0.8) between features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print(f"\nHigh correlations between features (>0.8):")
        for feat1, feat2, corr in high_corr_pairs[:10]:  # Show top 10
            print(f"  {feat1} <-> {feat2}: {corr:.4f}")
    else:
        print("\nNo high correlations between features found.")
    
    return {
        'target_correlations': dict(sorted_corrs),
        'high_feature_correlations': high_corr_pairs,
        'correlation_matrix_shape': corr_matrix.shape
    }


def create_visualizations(X: pd.DataFrame, y: pd.Series, output_dir: str = "plots"):
    """Create visualizations for feature analysis."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n=== Creating Visualizations (saved to {output_dir}/) ===")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Target distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        target_counts = y.value_counts().sort_index()
        class_names = ['Home Win', 'Draw', 'Away Win']
        
        bars = ax.bar(class_names, target_counts.values)
        ax.set_title('Target Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, count in zip(bars, target_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance (based on correlation with target)
        numeric_features = X.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 0:
            target_corrs = {}
            for col in numeric_features.columns:
                try:
                    corr = np.corrcoef(numeric_features[col].fillna(0), y)[0, 1]
                    if not np.isnan(corr):
                        target_corrs[col] = abs(corr)
                except:
                    continue
            
            if target_corrs:
                # Plot top 15 features
                sorted_corrs = sorted(target_corrs.items(), key=lambda x: x[1], reverse=True)[:15]
                features, correlations = zip(*sorted_corrs)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                bars = ax.barh(range(len(features)), correlations)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.set_xlabel('Absolute Correlation with Target')
                ax.set_title('Top 15 Features by Correlation with Target', fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for i, (bar, corr) in enumerate(zip(bars, correlations)):
                    width = bar.get_width()
                    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                           f'{corr:.3f}', ha='left', va='center', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Missing data heatmap
        missing_data = X.isnull()
        if missing_data.any().any():
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Only show columns with missing data
            missing_cols = missing_data.columns[missing_data.any()].tolist()
            if len(missing_cols) > 20:  # Limit to prevent overcrowding
                missing_cols = missing_cols[:20]
            
            sns.heatmap(missing_data[missing_cols].iloc[:1000], 
                       cbar=True, ax=ax, cmap='viridis')
            ax.set_title('Missing Data Pattern (First 1000 Samples)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Features')
            ax.set_ylabel('Samples')
            
            plt.tight_layout()
            plt.savefig(output_path / 'missing_data_pattern.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Feature distribution plots (top 6 features)
        if len(numeric_features.columns) > 0 and target_corrs:
            top_features = [feat for feat, _ in sorted_corrs[:6]]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, feature in enumerate(top_features):
                if feature in numeric_features.columns:
                    data = numeric_features[feature].dropna()
                    axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{feature}', fontweight='bold')
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(alpha=0.3)
            
            # Hide empty subplots
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_path / 'feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
        
    except ImportError:
        print("Matplotlib/Seaborn not available. Skipping visualizations.")
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")


def benchmark_performance(func, *args, **kwargs):
    """Benchmark function performance."""
    print(f"\nBenchmarking {func.__name__}...")
    
    # Memory before
    mem_before = get_memory_usage()
    
    # Time execution
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    # Memory after
    mem_after = get_memory_usage()
    
    execution_time = end_time - start_time
    memory_used = mem_after - mem_before
    
    print(f"  Execution time: {execution_time:.2f} seconds")
    print(f"  Memory used: {memory_used:.2f} MB")
    
    return result, {
        'execution_time': execution_time,
        'memory_used': memory_used,
        'memory_before': mem_before,
        'memory_after': mem_after
    }


def validate_feature_consistency(X: pd.DataFrame, meta: pd.DataFrame) -> Dict[str, Any]:
    """Validate feature consistency and data integrity."""
    print("\n=== Feature Validation ===")
    
    issues = []
    
    # Check for infinite values using a more robust approach
    inf_cols = []
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    print(f"Checking {len(numeric_cols)} numeric columns for validation issues...")
    
    for col in numeric_cols:
        try:
            # Method 1: Use pandas isin method (most reliable)
            has_inf = X[col].isin([np.inf, -np.inf]).sum() > 0
            if has_inf:
                inf_cols.append(col)
                issues.append(f"Infinite values in {col}")
        except Exception:
            try:
                # Method 2: Convert to numpy array first
                values = X[col].values
                if np.any(np.isinf(values)):
                    inf_cols.append(col)
                    issues.append(f"Infinite values in {col}")
            except Exception:
                # Method 3: Skip problematic columns
                print(f"Warning: Could not check infinite values in {col} - skipping")
                continue
    
    # Check for constant features
    constant_cols = []
    for col in numeric_cols:
        try:
            unique_count = X[col].nunique()
            if unique_count <= 1:
                constant_cols.append(col)
                issues.append(f"Constant feature: {col}")
        except Exception:
            print(f"Warning: Could not check uniqueness in {col} - skipping")
            continue
    
    # Check for features with excessive missing values (>95%)
    high_missing_cols = []
    for col in X.columns:
        try:
            missing_pct = X[col].isnull().sum() / len(X)
            if missing_pct > 0.95:
                high_missing_cols.append(col)
                issues.append(f"High missing values (>{missing_pct:.1%}) in {col}")
        except Exception:
            continue
    
    # Simplified duplicate check (only for small datasets to avoid performance issues)
    duplicate_pairs = []
    if len(numeric_cols) < 50:  # Only check if manageable number of columns
        for i, col1 in enumerate(numeric_cols[:20]):  # Limit to first 20 columns
            for col2 in numeric_cols[i+1:min(i+21, len(numeric_cols))]:  # Check against next 20
                try:
                    # Use correlation instead of exact equality (more robust)
                    corr = X[col1].corr(X[col2])
                    if abs(corr) > 0.999:  # Nearly identical
                        duplicate_pairs.append((col1, col2))
                        issues.append(f"Nearly duplicate features: {col1} and {col2} (corr={corr:.4f})")
                except Exception:
                    continue
    
    # Check date consistency
    if 'date' in meta.columns:
        date_issues = []
        try:
            dates = pd.to_datetime(meta['date'], errors='coerce')
            null_dates = dates.isnull().sum()
            if null_dates > 0:
                date_issues.append(f"Invalid dates found: {null_dates}")
            
            # Check for future dates (with some tolerance)
            max_date = dates.max()
            if pd.notna(max_date) and max_date > pd.Timestamp.now():
                date_issues.append("Future dates found")
            
            issues.extend(date_issues)
        except Exception as e:
            print(f"Warning: Could not validate dates: {e}")
    
    validation_results = {
        'infinite_values': inf_cols,
        'constant_features': constant_cols,
        'high_missing_features': high_missing_cols,
        'duplicate_features': duplicate_pairs,
        'all_issues': issues
    }
    
    print(f"Validation completed:")
    print(f"  - Infinite values: {len(inf_cols)} columns")
    print(f"  - Constant features: {len(constant_cols)} columns")
    print(f"  - High missing values: {len(high_missing_cols)} columns")
    print(f"  - Near-duplicate features: {len(duplicate_pairs)} pairs")
    
    if issues:
        print("\nTop validation issues:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more issues")
    else:
        print("✓ No major validation issues found!")
    
    return validation_results


def main():
    """Main testing function with comprehensive analysis."""
    print("="*70)
    print("ENHANCED FEATURE ENGINEERING TEST & ANALYSIS")
    print("="*70)
    
    # Initial memory usage
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    try:
        # Step 1: Fetch data
        print("\n" + "="*50)
        print("STEP 1: DATA LOADING")
        print("="*50)
        
        df, fetch_metrics = benchmark_performance(fetch_matches)
        print(f"Fetched {len(df):,} matches")
        
        # Analyze raw data
        raw_analysis = analyze_data_quality(df, "Raw Match Data")
        
        # Step 2: Build features
        print("\n" + "="*50)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*50)
        
        # Test with different configurations
        configs = [
            {'windows': [5], 'include_h2h': False, 'name': 'Basic'},
            {'windows': [3, 5, 10], 'include_h2h': True, 'name': 'Full'}
        ]
        
        results = {}
        
        for config in configs:
            print(f"\nTesting {config['name']} configuration...")
            
            (X, y, meta), metrics = benchmark_performance(
                build_feature_matrix, 
                df, 
                windows=config['windows'],
                include_h2h=config['include_h2h']
            )
            
            print(f"✓ {config['name']} feature matrix: {X.shape}")
            
            results[config['name']] = {
                'X': X, 'y': y, 'meta': meta,
                'metrics': metrics,
                'config': config
            }
        
        # Step 3: Detailed analysis of best configuration
        print("\n" + "="*50)
        print("STEP 3: DETAILED ANALYSIS (Full Configuration)")
        print("="*50)
        
        best_result = results['Full']
        X, y, meta = best_result['X'], best_result['y'], best_result['meta']
        
        # Data quality analysis
        feature_analysis = analyze_data_quality(X, "Feature Matrix")
        target_analysis = analyze_data_quality(pd.DataFrame({'target': y}), "Target Variable")
        
        # Feature analysis
        feature_stats = analyze_features(X, y)
        
        # Correlation analysis
        correlation_analysis = analyze_correlations(X, y, top_n=15)
        
        # Feature validation
        validation_results = validate_feature_consistency(X, meta)
        
        # Step 4: Performance comparison
        print("\n" + "="*50)
        print("STEP 4: CONFIGURATION COMPARISON")
        print("="*50)
        
        print(f"{'Configuration':<15} {'Features':<10} {'Samples':<10} {'Time (s)':<10} {'Memory (MB)':<12}")
        print("-" * 60)
        
        for name, result in results.items():
            X_shape = result['X'].shape
            metrics = result['metrics']
            print(f"{name:<15} {X_shape[1]:<10} {X_shape[0]:<10} "
                  f"{metrics['execution_time']:<10.2f} {metrics['memory_used']:<12.2f}")
        
        # Step 5: Feature groups analysis
        print("\n" + "="*50)
        print("STEP 5: FEATURE GROUPS ANALYSIS")
        print("="*50)
        
        feature_groups = get_feature_importance_groups()
        available_features = set(X.columns)
        
        for group_name, group_features in feature_groups.items():
            available_in_group = [f for f in group_features if f in available_features]
            if available_in_group:
                print(f"\n{group_name.upper()} ({len(available_in_group)} features):")
                for feature in available_in_group[:5]:  # Show first 5
                    print(f"  - {feature}")
                if len(available_in_group) > 5:
                    print(f"  ... and {len(available_in_group) - 5} more")
        
        # Step 6: Create visualizations
        print("\n" + "="*50)
        print("STEP 6: VISUALIZATIONS")
        print("="*50)
        
        create_visualizations(X, y)
        
        # Step 7: Summary
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        
        final_memory = get_memory_usage()
        total_memory_used = final_memory - initial_memory
        
        print(f"✓ Successfully processed {len(df):,} matches")
        print(f"✓ Generated {X.shape[1]} features across {X.shape[0]} samples")
        print(f"✓ Target distribution: {dict(y.value_counts().sort_index())}")
        print(f"✓ Missing data: {X.isnull().sum().sum()} total missing values")
        print(f"✓ Memory usage: {total_memory_used:.2f} MB total")
        
        if validation_results['all_issues']:
            print(f"⚠ Found {len(validation_results['all_issues'])} validation issues")
        else:
            print("✓ All validation checks passed")
        
        # Export feature information
        feature_info = {
            'feature_names': X.columns.tolist(),
            'feature_count': len(X.columns),
            'sample_count': len(X),
            'target_distribution': y.value_counts().to_dict(),
            'missing_values': X.isnull().sum().to_dict(),
            'feature_groups': {k: [f for f in v if f in X.columns] 
                             for k, v in feature_groups.items()},
            'data_types': X.dtypes.astype(str).to_dict(),
            'validation_issues': validation_results['all_issues']
        }
        
        import json
        with open('feature_analysis_report.json', 'w') as f:
            json.dump(feature_info, f, indent=2, default=str)
        
        print(f"✓ Feature analysis report saved to 'feature_analysis_report.json'")
        
        return X, y, meta, feature_info
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    X, y, meta, report = main()