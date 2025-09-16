# -- import packages: ---------------------------------------------------------
import pandas as pd
import numpy as np
import warnings

# -- set type hints: ----------------------------------------------------------
from typing import Tuple, Union


# -- operational cls: ---------------------------------------------------------
class CrossEntropy:
    """
    Cross-entropy calculator for cell fate prediction.

    This class calculates the cross-entropy between observed and predicted
    cell fate distributions, with proper handling of edge cases and numerical
    stability considerations.

    Attributes:
        epsilon (float): Small value to prevent log(0) numerical issues.

    Example:
        >>> ce_calc = CrossEntropy(epsilon=1e-8)
        >>> cross_entropy_value = ce_calc(observed_fates, predicted_fates)
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize the CrossEntropy calculator.

        Args:
            epsilon: Small value to prevent log(0). Should be small enough to not
                    affect valid probabilities but large enough to avoid extreme
                    negative log values. Default 1e-8 provides good balance.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if epsilon > 0.01:
            warnings.warn("Large epsilon values may affect cross-entropy accuracy")

        self.epsilon = epsilon

    def _ensure_same_columns(
        self, F_obs: pd.DataFrame, F_hat: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ensure both dataframes have the same columns in the same order.

        Args:
            F_obs: Observed fate distributions
            F_hat: Predicted fate distributions

        Returns:
            Tuple of (F_obs, F_hat) with aligned columns
        """
        # Get union of all columns, sorted for consistency
        all_cols = sorted(set(F_obs.columns) | set(F_hat.columns))

        # Add missing columns with zeros
        for col in all_cols:
            if col not in F_obs.columns:
                F_obs[col] = 0.0
            if col not in F_hat.columns:
                F_hat[col] = 0.0

        # Reorder columns to match
        F_obs = F_obs[all_cols]
        F_hat = F_hat[all_cols]

        return F_obs, F_hat

    def _normalize_predictions(self, F_hat: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize predictions to valid probability distributions.

        This method:
        1. Ensures non-negative values
        2. Handles zero-sum rows with uniform distribution
        3. Normalizes rows to sum to 1
        4. Clips to avoid log(0) issues while maintaining valid probabilities

        Args:
            F_hat: Raw prediction DataFrame

        Returns:
            Normalized prediction DataFrame
        """
        # Ensure non-negative values
        F_hat = F_hat.clip(lower=0)

        # Check for rows with zero sums
        row_sums = F_hat.sum(axis=1)
        valid_rows = row_sums > 0

        if valid_rows.sum() < len(F_hat):
            warnings.warn(
                f"Found {(~valid_rows).sum()} rows with zero or negative sums in predictions"
            )

        # Only normalize rows with positive sums
        F_hat.loc[valid_rows] = F_hat.loc[valid_rows].div(row_sums[valid_rows], axis=0)

        # Clip to avoid log(0) issues while maintaining valid probabilities
        # Use slightly smaller upper bound to ensure renormalization doesn't exceed 1
        max_val = 1.0 - (F_hat.shape[1] - 1) * self.epsilon
        F_hat.loc[valid_rows] = F_hat.loc[valid_rows].clip(
            lower=self.epsilon, upper=max_val
        )

        # Final normalization to ensure exact probability distribution
        F_hat.loc[valid_rows] = F_hat.loc[valid_rows].div(
            F_hat.loc[valid_rows].sum(axis=1), axis=0
        )

        return F_hat

    def _normalize_observations(self, F_obs: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize observed fate distributions to valid probabilities.

        Args:
            F_obs: Observed fate distributions

        Returns:
            Normalized observation DataFrame
        """
        # Ensure non-negative values
        F_obs = F_obs.clip(lower=0)

        # Normalize to probabilities
        row_sums = F_obs.sum(axis=1)
        valid_rows = row_sums > 0

        if valid_rows.sum() < len(F_obs):
            warnings.warn(
                f"Found {(~valid_rows).sum()} rows with zero or negative sums in observations"
            )

        # Only normalize rows with positive sums
        F_obs.loc[valid_rows] = F_obs.loc[valid_rows].div(row_sums[valid_rows], axis=0)

        return F_obs

    def __call__(
        self,
        F_obs: pd.DataFrame,
        F_hat: pd.DataFrame,
        return_per_cell: bool = False,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Calculate cross-entropy between observed and predicted fate distributions.

        The cross-entropy is calculated as: CE = -∑(p_true * log(p_pred))
        where the sum is over all fate categories for each cell, then averaged
        across all valid cells.

        Args:
            F_obs: DataFrame with true fate proportions (rows=cells, cols=fates)
            F_hat: DataFrame with predicted fate proportions (rows=cells, cols=fates)
            return_per_cell: If True, return (mean_ce, per_cell_ce_array)

        Returns:
            float: Mean cross-entropy across all valid cells
            or tuple: (mean_ce, per_cell_ce_array) if return_per_cell=True

        Raises:
            ValueError: If inputs have incompatible shapes or no valid data
        """
        # Input validation
        if len(F_obs) != len(F_hat):
            raise ValueError(f"Mismatched number of rows: {len(F_obs)} vs {len(F_hat)}")

        if len(F_obs) == 0:
            raise ValueError("Empty input DataFrames")

        # Make copies to avoid modifying originals
        true = F_obs.copy()
        pred = F_hat.copy()

        # Preserve original index types instead of converting to string
        # Only align indices if they're actually mismatched
        if not true.index.equals(pred.index):
            # Reindex both to their intersection
            common_idx = true.index.intersection(pred.index)
            if len(common_idx) == 0:
                raise ValueError("No common indices between F_obs and F_hat")
            true = true.loc[common_idx]
            pred = pred.loc[common_idx]

        # Ensure same columns
        true, pred = self._ensure_same_columns(true, pred)

        # Normalize both distributions
        true = self._normalize_observations(true)
        pred = self._normalize_predictions(pred)

        # Filter out rows where predicted distribution sums to 0 (invalid cases)
        valid_rows = pred.sum(axis=1) > 0

        print(f"valid_rows: {valid_rows.sum()}")

        if valid_rows.sum() == 0:
            raise ValueError("No valid rows found for cross-entropy calculation")

        if valid_rows.sum() < len(true):
            warnings.warn(f"Filtered out {(~valid_rows).sum()} invalid rows")

        true_valid = true.loc[valid_rows]
        pred_valid = pred.loc[valid_rows]

        # Calculate cross-entropy: CE = -sum(p_true * log(p_pred))
        log_pred = np.log(pred_valid.values)
        ce_per_cell = -np.sum(true_valid.values * log_pred, axis=1)

        # Calculate mean cross-entropy
        mean_ce = ce_per_cell.mean()

        if return_per_cell:
            return mean_ce, ce_per_cell
        else:
            return mean_ce

    def random_baseline(self, n_fates: int) -> float:
        """
        Calculate the expected cross-entropy for random predictions.

        For uniform random predictions across n_fates, the expected cross-entropy
        is log(n_fates).

        Args:
            n_fates: Number of possible fates/categories

        Returns:
            float: Expected cross-entropy for random predictions
        """
        if n_fates <= 0:
            raise ValueError("Number of fates must be positive")
        return np.log(n_fates)


# -- api-facing function: -----------------------------------------------------
def cross_entropy(
    F_obs: pd.DataFrame,
    F_hat: pd.DataFrame,
    epsilon: float = 1e-8,
    return_per_cell: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Convenience function to calculate cross-entropy.

    Args:
        F_obs: DataFrame with true fate proportions (rows=cells, cols=fates)
        F_hat: DataFrame with predicted fate proportions (rows=cells, cols=fates)
        epsilon: Small value to prevent log(0), default 1e-8
        return_per_cell: If True, return (mean_ce, per_cell_ce_array)

    Returns:
        float: Mean cross-entropy across all valid cells
        or tuple: (mean_ce, per_cell_ce_array) if return_per_cell=True

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Create example data
        >>> true_fates = pd.DataFrame({
        ...     'neutrophil': [1.0, 0.0, 0.6],
        ...     'monocyte': [0.0, 1.0, 0.4]
        ... })
        >>> pred_fates = pd.DataFrame({
        ...     'neutrophil': [0.9, 0.1, 0.7],
        ...     'monocyte': [0.1, 0.9, 0.3]
        ... })
        >>>
        >>> ce = cross_entropy(true_fates, pred_fates)
        >>> print(f"Cross-entropy: {ce:.3f}")
        >>>
        >>> # Compare to random baseline
        >>> baseline = CrossEntropy().random_baseline(n_fates=2)
        >>> print(f"Random baseline: {baseline:.3f}")
    """
    calc = CrossEntropy(epsilon=epsilon)
    return calc(F_obs, F_hat, return_per_cell=return_per_cell)
