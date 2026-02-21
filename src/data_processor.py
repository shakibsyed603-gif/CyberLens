import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys
import glob
import joblib
import gc  # garbage collector

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_DIR, EXCLUDE_COLUMNS, PROCESSED_DATA_DIR, SCALER_PATH

# -- paths --------------------------------------------------------------------
CIC_DIR   = os.path.join(DATA_DIR, "cic_ids2017")
PARQUET_PATH = os.path.join(PROCESSED_DATA_DIR, "combined_dataset.parquet")

# -- KDD column names ----------------------------------------------------------
KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'Label', 'difficulty'
]

# -- CIC-IDS2017 canonical feature set (shared numeric features) ---------------
# These are the numeric network-flow features common across all CIC CSV files.
# We map them to names matching the KDD feature space where possible.
CIC_FEATURE_MAP = {
    ' Flow Duration':              'duration',
    ' Total Fwd Packets':          'count',
    ' Total Backward Packets':     'srv_count',
    ' Total Length of Fwd Packets':'src_bytes',
    ' Total Length of Bwd Packets':'dst_bytes',
    ' Fwd Packet Length Max':      'hot',
    ' Fwd Packet Length Min':      'wrong_fragment',
    ' Flow Bytes/s':               'serror_rate',
    ' Flow Packets/s':             'srv_serror_rate',
    ' Flow IAT Mean':              'rerror_rate',
    ' Flow IAT Std':               'srv_rerror_rate',
    ' Fwd IAT Total':              'same_srv_rate',
    ' Bwd IAT Total':              'diff_srv_rate',
    ' SYN Flag Count':             'dst_host_serror_rate',
    ' RST Flag Count':             'dst_host_srv_serror_rate',
    ' PSH Flag Count':             'dst_host_rerror_rate',
    ' ACK Flag Count':             'dst_host_srv_rerror_rate',
    ' URG Flag Count':             'urgent',
    ' Average Packet Size':        'dst_host_same_srv_rate',
    ' Avg Fwd Segment Size':       'dst_host_diff_srv_rate',
    ' Label':                      'Label',
}

# Attack labels in CIC dataset that map to anomaly = 1
CIC_NORMAL_LABELS = {'BENIGN', 'Benign', 'benign'}


class DataProcessor:
    """RAM-safe data processor supporting KDD Cup and CIC-IDS2017 datasets."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None

    # -------------------------------------------------------------------------
    # KDD loading
    # -------------------------------------------------------------------------
    def load_kdd_data(self, sample_fraction: float = 1.0) -> pd.DataFrame:
        """Load KDD Cup dataset files (Train + Test)."""
        print("[LOAD] Loading KDD Cup dataset...")
        data_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
        if not data_files:
            raise FileNotFoundError(f"No KDD .txt files found in {DATA_DIR}")

        frames = []
        for file in data_files:
            print(f"   -> {os.path.basename(file)}")
            df = pd.read_csv(file, header=None, names=KDD_COLUMNS,
                             dtype=str, low_memory=False)
            if sample_fraction < 1.0:
                df = df.sample(frac=sample_fraction, random_state=42)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        print(f"   [OK] KDD shape: {combined.shape}")
        return combined

    # -------------------------------------------------------------------------
    # CIC-IDS2017 loading - chunked to stay RAM-safe
    # -------------------------------------------------------------------------
    def load_cic_data_chunked(
        self,
        sample_fraction: float = 0.05,
        chunk_size: int = 50_000,
    ) -> pd.DataFrame:
        """
        Load CIC-IDS2017 CSVs in chunks and sample rows to control RAM.

        sample_fraction : fraction of each CSV to keep (default 5 %)
        chunk_size      : rows processed at a time  (default 50 000)
        """
        csv_files = glob.glob(os.path.join(CIC_DIR, "*.csv"))
        if not csv_files:
            print(f"[WARN] No CIC-IDS2017 CSVs found in {CIC_DIR} - skipping.")
            return pd.DataFrame()

        print(f"[LOAD] Loading CIC-IDS2017 ({len(csv_files)} files, "
              f"{sample_fraction*100:.0f}% sample per file, "
              f"chunk_size={chunk_size:,})...")

        all_frames = []
        for csv_path in csv_files:
            fname = os.path.basename(csv_path)
            file_frames = []
            try:
                reader = pd.read_csv(
                    csv_path,
                    chunksize=chunk_size,
                    low_memory=False,
                    encoding='utf-8',
                    on_bad_lines='skip',
                )
                for chunk in reader:
                    # Keep only columns we care about
                    keep_cols = [c for c in CIC_FEATURE_MAP if c in chunk.columns]
                    if not keep_cols:
                        continue
                    chunk = chunk[keep_cols].copy()
                    # Sample fraction
                    n = max(1, int(len(chunk) * sample_fraction))
                    chunk = chunk.sample(n=n, random_state=42)
                    file_frames.append(chunk)

                if file_frames:
                    file_df = pd.concat(file_frames, ignore_index=True)
                    all_frames.append(file_df)
                    total_rows = sum(len(f) for f in file_frames)
                    print(f"   -> {fname}: kept {total_rows:,} rows")

            except Exception as e:
                print(f"   [WARN] Skipping {fname}: {e}")
            finally:
                gc.collect()

        if not all_frames:
            print("[WARN] No CIC data could be loaded.")
            return pd.DataFrame()

        cic_df = pd.concat(all_frames, ignore_index=True)
        gc.collect()

        # Rename columns to match KDD naming
        cic_df = cic_df.rename(columns=CIC_FEATURE_MAP)

        # Keep only columns that are in the map values
        valid_cols = list(set(CIC_FEATURE_MAP.values()) & set(cic_df.columns))
        cic_df = cic_df[valid_cols]

        print(f"   [OK] CIC-IDS2017 shape after sampling: {cic_df.shape}")
        return cic_df

    # -------------------------------------------------------------------------
    # Label helpers
    # -------------------------------------------------------------------------
    def create_binary_labels(self, df: pd.DataFrame, source: str = 'kdd') -> pd.DataFrame:
        """Create is_anomaly column (0 = normal, 1 = attack)."""
        if source == 'kdd':
            df['is_anomaly'] = (df['Label'].astype(str).str.strip() != 'normal').astype(int)
        else:  # CIC
            df['is_anomaly'] = (~df['Label'].astype(str).str.strip().isin(CIC_NORMAL_LABELS)).astype(int)

        print(f"   Normal : {(df['is_anomaly'] == 0).sum():,}")
        print(f"   Anomaly: {(df['is_anomaly'] == 1).sum():,}")
        return df

    # -------------------------------------------------------------------------
    # Cleaning
    # -------------------------------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean, encode, and drop unwanted columns."""
        print("[CLEAN] Cleaning data...")

        # Convert everything numeric where possible
        for col in df.columns:
            if col not in ('Label', 'is_anomaly'):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Replace inf
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop columns with > 50% NaN
        miss = df.isnull().mean()
        bad_cols = miss[miss > 0.5].index.tolist()
        if bad_cols:
            print(f"   Dropping >50% NaN columns: {bad_cols}")
            df = df.drop(columns=bad_cols)

        # Fill remaining NaN with column median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Encode any remaining categoricals (except Label / is_anomaly)
        cat_cols = [c for c in df.select_dtypes(include='object').columns
                    if c not in ('Label', 'is_anomaly')]
        for col in cat_cols:
            df[col] = pd.Categorical(df[col]).codes

        # Drop excluded columns (Label, difficulty, etc.) but keep is_anomaly
        drop_cols = [c for c in EXCLUDE_COLUMNS if c in df.columns and c != 'is_anomaly']
        df = df.drop(columns=drop_cols, errors='ignore')

        print(f"   [OK] Cleaned shape: {df.shape}")
        return df

    # -------------------------------------------------------------------------
    # Feature preparation
    # -------------------------------------------------------------------------
    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = True):
        """Scale features and return (X_scaled, y)."""
        print("[PREP] Preparing features...")
        X = df.drop('is_anomaly', axis=1)
        y = df['is_anomaly']
        self.feature_columns = X.columns.tolist()

        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
            joblib.dump(self.scaler, SCALER_PATH)
            print(f"   Scaler saved -> {SCALER_PATH}")
        else:
            X_scaled = self.scaler.transform(X)

        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        return X_scaled, y

    # -------------------------------------------------------------------------
    # Save / Load parquet cache
    # -------------------------------------------------------------------------
    def save_parquet(self, df: pd.DataFrame, path: str = PARQUET_PATH):
        """Save dataframe as compressed parquet (~10× smaller than CSV)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False, compression='snappy')
        size_mb = os.path.getsize(path) / 1_048_576
        print(f"   [SAVE] Saved parquet: {path}  ({size_mb:.1f} MB)")

    def load_parquet(self, path: str = PARQUET_PATH) -> pd.DataFrame:
        """Load the pre-processed parquet cache."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No parquet cache found at {path}. "
                                    "Run process_full_pipeline() first.")
        print(f"[CACHE] Loading cached parquet from {path} ...")
        df = pd.read_parquet(path)
        print(f"   [OK] Loaded shape: {df.shape}")
        return df

    # -------------------------------------------------------------------------
    # Full combined pipeline
    # -------------------------------------------------------------------------
    def process_full_pipeline(
        self,
        kdd_sample_fraction: float = 1.0,
        cic_sample_fraction: float = 0.05,
        cic_chunk_size: int = 50_000,
        use_cache: bool = True,
        rebuild_cache: bool = False,
    ):
        """
        Smart pipeline that:
          1. Loads KDD + CIC-IDS2017 safely (chunked + sampled).
          2. Aligns columns between the two datasets.
          3. Cleans and scales features.
          4. Saves a small compressed .parquet cache.
          5. Returns (X_scaled, y).

        Params
        ------
        kdd_sample_fraction : fraction of KDD to use  (1.0 = all, ~22 MB)
        cic_sample_fraction : fraction of CIC to use  (0.05 = 5%, ~42 MB RAM)
        cic_chunk_size      : rows per chunk when reading CIC CSVs
        use_cache           : if True, load from parquet cache if it exists
        rebuild_cache       : if True, ignore existing cache and reprocess
        """

        # ── Try cache first ───────────────────────────────────────────────
        if use_cache and not rebuild_cache and os.path.exists(PARQUET_PATH):
            print("[OK] Found existing parquet cache - loading directly!")
            df = self.load_parquet()
            X, y = self.prepare_features(df, fit_scaler=True)
            return X, y

        print("=" * 60)
        print("[PILOT] CyberLens - Smart Data Pipeline")
        print("=" * 60)

        # ── Load KDD ──────────────────────────────────────────────────────
        kdd_df = self.load_kdd_data(sample_fraction=kdd_sample_fraction)
        kdd_df = self.create_binary_labels(kdd_df, source='kdd')
        kdd_df = self.clean_data(kdd_df)
        gc.collect()

        # ── Load CIC ──────────────────────────────────────────────────────
        cic_df = self.load_cic_data_chunked(
            sample_fraction=cic_sample_fraction,
            chunk_size=cic_chunk_size,
        )

        if not cic_df.empty:
            cic_df = self.create_binary_labels(cic_df, source='cic')
            cic_df = self.clean_data(cic_df)
            gc.collect()

            # ── Align columns: keep only shared features ──────────────────
            shared_cols = list(
                (set(kdd_df.columns) & set(cic_df.columns)) | {'is_anomaly'}
            )
            kdd_df  = kdd_df[[c for c in shared_cols if c in kdd_df.columns]]
            cic_df  = cic_df[[c for c in shared_cols if c in cic_df.columns]]

            combined = pd.concat([kdd_df, cic_df], ignore_index=True)
            del kdd_df, cic_df
            gc.collect()
            print(f"\n[MERGE] Combined dataset shape: {combined.shape}")
        else:
            combined = kdd_df
            del kdd_df
            gc.collect()
            print(f"\n[WARN] Using KDD only. Shape: {combined.shape}")

        # ── Final clean pass on combined ──────────────────────────────────
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

        # ── Save parquet cache ────────────────────────────────────────────
        print("\n[SAVE] Saving compressed parquet cache...")
        self.save_parquet(combined)

        # ── Feature preparation ───────────────────────────────────────────
        X, y = self.prepare_features(combined, fit_scaler=True)
        del combined
        gc.collect()

        # ── Save legacy pkl for backward-compat ───────────────────────────
        processed_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
        joblib.dump({'X': X, 'y': y, 'feature_columns': self.feature_columns}, processed_path)
        print(f"   Legacy pkl saved -> {processed_path}")

        print("\n[DONE] Pipeline complete!")
        print(f"   Features : {X.shape[1]}")
        print(f"   Samples  : {X.shape[0]:,}")
        print(f"   Anomaly% : {y.mean()*100:.1f}%")
        return X, y


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    processor = DataProcessor()
    X, y = processor.process_full_pipeline(
        kdd_sample_fraction=1.0,   # use all of KDD (it's small, ~22 MB)
        cic_sample_fraction=0.05,  # use 5% of CIC  (~42 MB RAM safe)
        cic_chunk_size=50_000,
        use_cache=False,           # force rebuild on first run
        rebuild_cache=True,
    )
    print(f"\nReady for training: X={X.shape}, y={y.shape}")
