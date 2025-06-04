"""
Data Splitter for Out-of-Sample Testing

This module provides functionality to split historical data into training, validation,
and test sets for robust out-of-sample testing of trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

try:
    from ..utils.logger import get_logger
except ImportError:
    from src.utils.logger import get_logger


@dataclass
class DataSplit:
    """Container for split data sets."""
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    split_info: Dict[str, any]
    
    def __post_init__(self):
        """Validate the split after initialization."""
        total_original = self.split_info.get('original_length', 0)
        total_split = len(self.train) + len(self.validation) + len(self.test)
        
        if total_split != total_original:
            warnings.warn(f"Data length mismatch: original={total_original}, split={total_split}")
    
    @property
    def train_ratio(self) -> float:
        """Get actual training data ratio."""
        total = len(self.train) + len(self.validation) + len(self.test)
        return len(self.train) / total if total > 0 else 0.0
    
    @property
    def validation_ratio(self) -> float:
        """Get actual validation data ratio."""
        total = len(self.train) + len(self.validation) + len(self.test)
        return len(self.validation) / total if total > 0 else 0.0
    
    @property
    def test_ratio(self) -> float:
        """Get actual test data ratio."""
        total = len(self.train) + len(self.validation) + len(self.test)
        return len(self.test) / total if total > 0 else 0.0
    
    def summary(self) -> Dict[str, any]:
        """Get summary of the data split."""
        return {
            'train_periods': len(self.train),
            'validation_periods': len(self.validation),
            'test_periods': len(self.test),
            'train_ratio': self.train_ratio,
            'validation_ratio': self.validation_ratio,
            'test_ratio': self.test_ratio,
            'train_date_range': (self.train.index[0], self.train.index[-1]) if len(self.train) > 0 else None,
            'validation_date_range': (self.validation.index[0], self.validation.index[-1]) if len(self.validation) > 0 else None,
            'test_date_range': (self.test.index[0], self.test.index[-1]) if len(self.test) > 0 else None,
            'split_method': self.split_info.get('method', 'unknown'),
            'split_params': self.split_info.get('params', {})
        }


class DataSplitter:
    """
    Data splitter for out-of-sample testing.
    
    Provides multiple methods for splitting time series data into training,
    validation, and test sets while maintaining temporal integrity.
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_periods_per_set: int = 100,
        ensure_temporal_order: bool = True
    ):
        """
        Initialize the DataSplitter.
        
        Args:
            train_ratio: Proportion of data for training (0.0-1.0)
            validation_ratio: Proportion of data for validation (0.0-1.0)
            test_ratio: Proportion of data for testing (0.0-1.0)
            min_periods_per_set: Minimum number of periods required in each set
            ensure_temporal_order: Whether to maintain temporal order in splits
        """
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.min_periods_per_set = min_periods_per_set
        self.ensure_temporal_order = ensure_temporal_order
        
        # Validate ratios
        if not self._validate_ratios():
            raise ValueError("Invalid ratios: must sum to 1.0 and be between 0.0 and 1.0")
        
        self.logger = get_logger("data_splitter")
    
    def _validate_ratios(self) -> bool:
        """Validate that ratios are valid."""
        ratios = [self.train_ratio, self.validation_ratio, self.test_ratio]
        
        # Check individual ratios
        if any(r < 0.0 or r > 1.0 for r in ratios):
            return False
        
        # Check sum (allow small floating point errors)
        if abs(sum(ratios) - 1.0) > 1e-6:
            return False
        
        return True
    
    def chronological_split(
        self,
        data: pd.DataFrame,
        gap_days: int = 0
    ) -> DataSplit:
        """
        Split data chronologically (most common for time series).
        
        Args:
            data: DataFrame with datetime index
            gap_days: Number of days to skip between sets (to avoid look-ahead bias)
            
        Returns:
            DataSplit object containing train, validation, and test sets
        """
        if len(data) < self.min_periods_per_set * 3:
            raise ValueError(f"Insufficient data: need at least {self.min_periods_per_set * 3} periods")
        
        # Sort by index to ensure chronological order
        data_sorted = data.sort_index()
        
        # Calculate split points
        total_periods = len(data_sorted)
        train_end = int(total_periods * self.train_ratio)
        validation_end = int(total_periods * (self.train_ratio + self.validation_ratio))
        
        # Apply gaps if specified
        if gap_days > 0:
            gap_periods = self._days_to_periods(data_sorted, gap_days)
            train_end = max(train_end - gap_periods, self.min_periods_per_set)
            validation_end = min(validation_end + gap_periods, total_periods - self.min_periods_per_set)
        
        # Create splits
        train_data = data_sorted.iloc[:train_end].copy()
        validation_data = data_sorted.iloc[train_end:validation_end].copy()
        test_data = data_sorted.iloc[validation_end:].copy()
        
        # Validate minimum periods
        if len(train_data) < self.min_periods_per_set:
            raise ValueError(f"Training set too small: {len(train_data)} < {self.min_periods_per_set}")
        if len(validation_data) < self.min_periods_per_set:
            raise ValueError(f"Validation set too small: {len(validation_data)} < {self.min_periods_per_set}")
        if len(test_data) < self.min_periods_per_set:
            raise ValueError(f"Test set too small: {len(test_data)} < {self.min_periods_per_set}")
        
        split_info = {
            'method': 'chronological',
            'original_length': len(data),
            'gap_days': gap_days,
            'params': {
                'train_ratio': self.train_ratio,
                'validation_ratio': self.validation_ratio,
                'test_ratio': self.test_ratio
            }
        }
        
        self.logger.info(f"Chronological split: train={len(train_data)}, val={len(validation_data)}, test={len(test_data)}")
        
        return DataSplit(train_data, validation_data, test_data, split_info)
    
    def random_split(
        self,
        data: pd.DataFrame,
        random_seed: Optional[int] = None,
        block_size: Optional[int] = None
    ) -> DataSplit:
        """
        Split data randomly (useful for non-temporal validation).
        
        Args:
            data: DataFrame to split
            random_seed: Random seed for reproducibility
            block_size: Size of contiguous blocks to maintain some temporal structure
            
        Returns:
            DataSplit object containing train, validation, and test sets
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if len(data) < self.min_periods_per_set * 3:
            raise ValueError(f"Insufficient data: need at least {self.min_periods_per_set * 3} periods")
        
        total_periods = len(data)
        
        if block_size is None:
            # Simple random sampling
            indices = np.random.permutation(total_periods)
        else:
            # Block-based random sampling to maintain some temporal structure
            indices = self._block_random_sampling(total_periods, block_size)
        
        # Calculate split points
        train_end = int(total_periods * self.train_ratio)
        validation_end = int(total_periods * (self.train_ratio + self.validation_ratio))
        
        # Create splits using random indices
        train_indices = indices[:train_end]
        validation_indices = indices[train_end:validation_end]
        test_indices = indices[validation_end:]
        
        # Sort indices if temporal order should be maintained within each set
        if self.ensure_temporal_order:
            train_indices = np.sort(train_indices)
            validation_indices = np.sort(validation_indices)
            test_indices = np.sort(test_indices)
        
        train_data = data.iloc[train_indices].copy()
        validation_data = data.iloc[validation_indices].copy()
        test_data = data.iloc[test_indices].copy()
        
        split_info = {
            'method': 'random',
            'original_length': len(data),
            'random_seed': random_seed,
            'block_size': block_size,
            'params': {
                'train_ratio': self.train_ratio,
                'validation_ratio': self.validation_ratio,
                'test_ratio': self.test_ratio
            }
        }
        
        self.logger.info(f"Random split: train={len(train_data)}, val={len(validation_data)}, test={len(test_data)}")
        
        return DataSplit(train_data, validation_data, test_data, split_info)
    
    def walk_forward_split(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
        test_size_ratio: float = 0.2
    ) -> List[DataSplit]:
        """
        Create multiple walk-forward splits for time series cross-validation.
        
        Args:
            data: DataFrame to split
            n_splits: Number of walk-forward splits to create
            test_size_ratio: Proportion of data to use for testing in each split
            
        Returns:
            List of DataSplit objects
        """
        total_periods = len(data)
        min_total_needed = self.min_periods_per_set * 3 * n_splits  # Conservative estimate
        
        if total_periods < min_total_needed:
            # Reduce n_splits if data is insufficient
            max_possible_splits = total_periods // (self.min_periods_per_set * 3)
            if max_possible_splits < 1:
                raise ValueError(f"Insufficient data for walk-forward splits: need at least {self.min_periods_per_set * 3} periods")
            n_splits = min(n_splits, max_possible_splits)
            self.logger.warning(f"Reduced n_splits to {n_splits} due to insufficient data")
        
        splits = []
        test_size = max(int(total_periods * test_size_ratio), self.min_periods_per_set)
        
        # Calculate step size for walk-forward - ensure we can fit all splits
        available_periods = total_periods - test_size
        step_size = max(available_periods // n_splits, self.min_periods_per_set)
        
        for i in range(n_splits):
            # Calculate split boundaries
            test_start = min(step_size * (i + 1), total_periods - test_size)
            test_end = min(test_start + test_size, total_periods)
            
            # Skip if we can't create a valid test set
            if test_end - test_start < self.min_periods_per_set:
                continue
            
            # Training data: all data before test period
            train_end = test_start
            
            # Validation data: portion before test
            val_size = max(int(train_end * 0.15), self.min_periods_per_set)
            val_start = max(train_end - val_size, 0)
            
            # Ensure we have enough training data
            if val_start < self.min_periods_per_set:
                continue
            
            # Create splits
            train_data = data.iloc[:val_start].copy()
            validation_data = data.iloc[val_start:train_end].copy()
            test_data = data.iloc[test_start:test_end].copy()
            
            # Validate minimum periods
            if (len(train_data) >= self.min_periods_per_set and 
                len(validation_data) >= self.min_periods_per_set and 
                len(test_data) >= self.min_periods_per_set):
                
                split_info = {
                    'method': 'walk_forward',
                    'split_number': i + 1,
                    'total_splits': n_splits,
                    'original_length': len(data),
                    'params': {
                        'test_size_ratio': test_size_ratio
                    }
                }
                
                splits.append(DataSplit(train_data, validation_data, test_data, split_info))
        
        self.logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def _block_random_sampling(self, total_periods: int, block_size: int) -> np.ndarray:
        """Create random sampling with blocks to maintain some temporal structure."""
        n_blocks = total_periods // block_size
        remaining = total_periods % block_size
        
        # Create block indices
        block_indices = np.random.permutation(n_blocks)
        
        # Create full index array
        indices = []
        for block_idx in block_indices:
            start_idx = block_idx * block_size
            end_idx = start_idx + block_size
            indices.extend(range(start_idx, end_idx))
        
        # Add remaining indices
        if remaining > 0:
            remaining_indices = list(range(n_blocks * block_size, total_periods))
            np.random.shuffle(remaining_indices)
            indices.extend(remaining_indices)
        
        return np.array(indices)
    
    def _days_to_periods(self, data: pd.DataFrame, days: int) -> int:
        """Convert days to number of periods based on data frequency."""
        if len(data) < 2:
            return days  # Fallback
        
        # Estimate frequency from first few periods
        time_diff = data.index[1] - data.index[0]
        periods_per_day = timedelta(days=1) / time_diff
        
        return int(days * periods_per_day)
    
    def validate_split(self, split: DataSplit) -> Dict[str, bool]:
        """
        Validate a data split for common issues.
        
        Args:
            split: DataSplit object to validate
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Check for data leakage (overlapping indices)
        train_indices = set(split.train.index)
        val_indices = set(split.validation.index)
        test_indices = set(split.test.index)
        
        results['no_train_val_overlap'] = len(train_indices & val_indices) == 0
        results['no_train_test_overlap'] = len(train_indices & test_indices) == 0
        results['no_val_test_overlap'] = len(val_indices & test_indices) == 0
        
        # Check minimum periods
        results['train_min_periods'] = len(split.train) >= self.min_periods_per_set
        results['val_min_periods'] = len(split.validation) >= self.min_periods_per_set
        results['test_min_periods'] = len(split.test) >= self.min_periods_per_set
        
        # Check temporal order (if required)
        if self.ensure_temporal_order:
            results['train_temporal_order'] = split.train.index.is_monotonic_increasing
            results['val_temporal_order'] = split.validation.index.is_monotonic_increasing
            results['test_temporal_order'] = split.test.index.is_monotonic_increasing
        
        # Check for missing data
        results['train_no_missing'] = not split.train.isnull().any().any()
        results['val_no_missing'] = not split.validation.isnull().any().any()
        results['test_no_missing'] = not split.test.isnull().any().any()
        
        # Overall validation
        results['overall_valid'] = all(results.values())
        
        return results
    
    def get_split_statistics(self, split: DataSplit) -> Dict[str, any]:
        """Get detailed statistics about a data split."""
        stats = split.summary()
        
        # Add data quality metrics
        for name, data in [('train', split.train), ('validation', split.validation), ('test', split.test)]:
            if len(data) > 0:
                stats[f'{name}_missing_pct'] = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
                
                # Price statistics (assuming 'close' column exists)
                if 'close' in data.columns:
                    returns = data['close'].pct_change().dropna()
                    stats[f'{name}_return_mean'] = returns.mean()
                    stats[f'{name}_return_std'] = returns.std()
                    stats[f'{name}_return_skew'] = returns.skew()
                    stats[f'{name}_return_kurt'] = returns.kurtosis()
        
        return stats 