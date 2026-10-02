"""
Outlier handling utilities for CLIF donor identifier.

This module provides functions to detect and handle outliers in clinical data
based on configurable range specifications from outlier_config.yaml.
Values outside the specified ranges are converted to null.
"""

import yaml
import polars as pl
from typing import Optional, Dict, Any
from pathlib import Path


def apply_outlier_handling(
    df: pl.DataFrame,
    table_name: str,
    outlier_config_path: Optional[str] = None
) -> pl.DataFrame:
    """
    Apply outlier handling to a Polars DataFrame.

    Values outside acceptable ranges are converted to null.
    For category-dependent columns (vitals, labs, assessments), ranges
    are applied based on the category value.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame to process
    table_name : str
        Name of the table (must match key in outlier_config.yaml)
    outlier_config_path : str, optional
        Path to outlier configuration YAML.
        If None, uses 'config/outlier_config.yaml'

    Returns
    -------
    pl.DataFrame
        DataFrame with outliers replaced by null values

    Examples
    --------
    >>> vitals_df = apply_outlier_handling(vitals_df, 'vitals')
    >>> labs_df = apply_outlier_handling(labs_df, 'labs')
    >>> resp_df = apply_outlier_handling(resp_df, 'respiratory_support')
    """
    # Load configuration
    config = _load_outlier_config(outlier_config_path)
    if not config:
        print(f"Warning: Could not load outlier config")
        return df

    # Get table-specific configuration
    table_config = config.get('tables', {}).get(table_name, {})
    if not table_config:
        print(f"No outlier configuration found for table: {table_name}")
        return df

    # Filter to columns that exist in the dataframe
    existing_columns = {
        col: conf for col, conf in table_config.items()
        if col in df.columns
    }

    if not existing_columns:
        print(f"No configured columns found in dataframe for table: {table_name}")
        return df

    print(f"\nApplying outlier thresholds for table: {table_name}")

    # Build all column expressions
    expressions = []
    for column_name, column_config in existing_columns.items():
        expr = _build_column_expression(df, table_name, column_name, column_config)
        if expr is not None:
            expressions.append(expr)

    # Apply all transformations at once
    if expressions:
        df = df.with_columns(expressions)
        print(f"✓ Applied outlier handling to {len(expressions)} column(s)")

    return df


def outlier_range(
    table_name: str,
    column_name: str,
    category: Optional[str] = None,
    outlier_config_path: Optional[str] = None
) -> tuple:
    """Return (min, max) for one column, or for one category within it.

    Lets a query apply a bound inline while outlier_config.yaml stays the only
    place the number is written down.

        >>> outlier_range('vitals', 'vital_value', 'weight_kg')
        (20.0, 1100.0)

    Raises rather than returning a default: a missing entry means a range was
    renamed or deleted, and silently applying no bound is how bad values reach
    the cohort.
    """
    config = _load_outlier_config(outlier_config_path)
    if config is None:
        raise FileNotFoundError("outlier_config.yaml could not be read")
    spec = config['tables'][table_name][column_name]
    if category is not None:
        spec = spec[category]
    return float(spec['min']), float(spec['max'])


def _load_outlier_config(config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load outlier configuration from YAML file."""
    try:
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'outlier_config.yaml'

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    except Exception as e:
        print(f"Error loading outlier configuration: {str(e)}")
        return None


def _build_column_expression(
    df: pl.DataFrame,
    table_name: str,
    column_name: str,
    column_config: Dict[str, Any]
):
    """Build a Polars expression for a column based on its configuration."""

    # Category-dependent columns (vitals, labs, patient_assessments)
    if table_name in ['vitals', 'labs', 'patient_assessments']:
        if column_name in ['vital_value', 'lab_value_numeric', 'numerical_value']:
            return _build_category_dependent_expression(
                df, table_name, column_name, column_config
            )

    # Medication dose column
    elif table_name == 'medication_admin_continuous' and column_name == 'med_dose':
        return _build_medication_expression(column_config)

    # Simple range columns (default)
    return _build_simple_range_expression(column_name, column_config)


def _build_category_dependent_expression(
    df: pl.DataFrame,
    table_name: str,
    column_name: str,
    column_config: Dict[str, Any]
):
    """Build expression for category-dependent columns like vitals and labs."""

    # Determine category column name
    category_mapping = {
        'vitals': 'vital_category',
        'labs': 'lab_category',
        'patient_assessments': 'assessment_category'
    }

    category_col = category_mapping.get(table_name)
    if not category_col or category_col not in df.columns:
        print(f"Warning: Category column '{category_col}' not found. Skipping {column_name}.")
        return None

    # Start with the original column value
    expr = pl.col(column_name)

    # Build chained when-then-otherwise for each category
    for category, range_config in column_config.items():
        if isinstance(range_config, dict) and 'min' in range_config and 'max' in range_config:
            min_val = range_config['min']
            max_val = range_config['max']

            # Condition: category matches AND value is outlier
            condition = (
                (pl.col(category_col).str.to_lowercase() == category.lower()) &
                ((pl.col(column_name) < min_val) | (pl.col(column_name) > max_val))
            )

            # Replace outliers with null
            expr = pl.when(condition).then(None).otherwise(expr)

    return expr.alias(column_name)


def _build_medication_expression(column_config: Dict[str, Any]):
    """Build expression for medication dose column."""

    expr = pl.col('med_dose')

    # Build chained when-then-otherwise for each medication/unit combo
    for med_category, unit_configs in column_config.items():
        if isinstance(unit_configs, dict):
            for unit, range_config in unit_configs.items():
                if isinstance(range_config, dict) and 'min' in range_config and 'max' in range_config:
                    min_val = range_config['min']
                    max_val = range_config['max']

                    # Condition: medication/unit matches AND value is outlier
                    condition = (
                        (pl.col('med_category').str.to_lowercase() == med_category.lower()) &
                        (pl.col('med_dose_unit').str.to_lowercase() == unit.lower()) &
                        ((pl.col('med_dose') < min_val) | (pl.col('med_dose') > max_val))
                    )

                    # Replace outliers with null
                    expr = pl.when(condition).then(None).otherwise(expr)

    return expr.alias('med_dose')


def _build_simple_range_expression(column_name: str, column_config: Dict[str, Any]):
    """Build expression for simple range columns."""

    if isinstance(column_config, dict) and 'min' in column_config and 'max' in column_config:
        min_val = column_config['min']
        max_val = column_config['max']

        # Simple outlier condition
        expr = pl.when(
            (pl.col(column_name) < min_val) | (pl.col(column_name) > max_val)
        ).then(None).otherwise(pl.col(column_name))

        return expr.alias(column_name)

    return None
