import time
import os
import polars as pl


def read_data(filepath, filetype, filter_ids=None, id_column=None):
      """
      Read data from file based on file type with ID filtering BEFORE loading to memory.
      
      Parameters:
          filepath (str): Path to the file.
          filetype (str): Type of the file ('csv' or 'parquet').
          filter_ids (list or set, optional): List/set of IDs to filter by.
          id_column (str, optional): Column name to filter on. Required if filter_ids provided.
      
      Returns:
          DataFrame: Filtered DataFrame loaded only for matching IDs.
      """
      start_time = time.time()
      file_name = os.path.basename(filepath)

      if filter_ids is not None and id_column is None:
          raise ValueError("id_column must be specified when filter_ids is provided.")

      # Use lazy evaluation - filter happens BEFORE loading to memory
      if filetype == 'parquet':
          df = pl.scan_parquet(filepath)
      elif filetype == 'csv':
          df = pl.scan_csv(filepath)
      else:
          raise ValueError("Unsupported file type. Please provide either 'csv' or 'parquet'.")

      # Apply filter lazily (executed on disk, not in memory)
      if filter_ids is not None:
          if isinstance(filter_ids, list):
              filter_ids = set(filter_ids)
          df = df.filter(pl.col(id_column).is_in(filter_ids))

      # NOW collect/materialize only the filtered rows
      df = df.collect()

      end_time = time.time()
      load_time = end_time - start_time
      dataset_size_mb = df.estimated_size() / (1024 * 1024)

      print(f"File name: {file_name}")
      print(f"Time taken to load the dataset: {load_time:.2f} seconds")
      print(f"Size of the loaded dataset: {dataset_size_mb:.2f} MB\n")

      return df