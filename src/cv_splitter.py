import numpy as np

class PurgedEmbargoTimeSeriesCV:
    def __init__(self, n_splits=5, purge_window=5, embargo_window=21):
        self.n_splits = n_splits
        self.purge_window = purge_window     
        self.embargo_window = embargo_window 

    def split(self, X):
        n_samples = len(X)
        fold_bounds = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)
        for i in range(self.n_splits):
            val_start = fold_bounds[i]
            val_end = fold_bounds[i + 1]
            val_indices = np.arange(val_start, val_end)
            
            train_pre_end = max(0, val_start - self.purge_window)
            train_pre_indices = np.arange(0, train_pre_end)
            
            train_post_start = min(n_samples, val_end + self.embargo_window)
            train_post_indices = np.arange(train_post_start, n_samples)
            
            train_indices = np.concatenate([train_pre_indices, train_post_indices])
            if len(train_indices) == 0:
                continue
            yield train_indices, val_indices