from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from config import RAW_DIR, PROCESSED_DIR

# 1. LOAD GEO DATASET

def load_geo_series_matrix(filename: str):
    file_path = RAW_DIR / filename

    with open(file_path, "r") as f:
        lines = f.readlines()

    return lines

# 2. EXTRACTING SAMPLE INFORMATION (IDs, Titles, Disease Status)

def extract_sample_info(lines):
    IDs, Titles, Status = [], [], []

    for line in lines:
        if line.startswith("!Sample_geo_accession"):
            IDs = line.strip().split("\t")[1:]

        if line.startswith("!Sample_title"):
            Titles = line.strip().split("\t")[1:]

        if line.startswith("!Sample_characteristics_ch1"):
            if "disease" in line.lower():
                chars = line.strip().split("\t")[1:]
                Status = [1 if "ankylosing" in c.lower() else 0 for c in chars]

    sample_info = pd.DataFrame({
        "sample_id": IDs,
        "title": Titles,
        "disease_status": Status
    })

    sample_info.columns = sample_info.columns.str.strip().str.lower()

    return sample_info


# 3. BUILDING EXPRESSION MATRIX

def build_expression_matrix(lines):
    start, end = None, None

    for i, line in enumerate(lines):
        if line.startswith("!series_matrix_table_begin"):
            start = i + 1
        if line.startswith("!series_matrix_table_end"):
            end = i
            break

    data_lines = lines[start:end]

    header = data_lines[0].strip().replace('"', "").split("\t")

    rows = [line.strip().split("\t") for line in data_lines[1:]]

    df = pd.DataFrame(rows, columns=header)
    df.set_index("ID_REF", inplace=True)

    df = df.apply(pd.to_numeric, errors="coerce")

    return df


# 4. QC + NORMALIZATION

def preprocess_expression(df):
    # missing filter
    missing_pct = df.isnull().sum(axis=1) / df.shape[1]
    df = df[missing_pct < 0.2]
    # imputation
    df = df.fillna(df.median(axis=1), axis=0)
    # variance filter
    var = df.var(axis=1)
    df = df[var > var.quantile(0.25)]
    # z-score normalization (per gene)
    df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    return df

# 5. BUILDING ML DATASET

def build_ml_dataset(expression_df, sample_info):

    ml_data = expression_df.T.copy()

    ml_data.index.name = "sample_id"
    ml_data = ml_data.reset_index()

    sample_info["sample_id"] = sample_info["sample_id"].astype(str)
    ml_data["sample_id"] = ml_data["sample_id"].astype(str)

    ml_data = ml_data.merge(
        sample_info[["sample_id", "disease_status"]],
        on="sample_id",
        how="inner"
    )

    ml_data = ml_data.set_index("sample_id")

    return ml_data



# 6. DIMENSIONALITY REDUCTION ANALYSIS(PCA & LDA)

def run_pca_lda(ml_data):
    X = ml_data.drop("disease_status", axis=1)
    y = ml_data["disease_status"]

    # PCA
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)

    # LDA
    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X, y)

    return pca, X_pca, lda, X_lda


# 7. SAVE OUTPUT

def save_dataset(ml_data, filename="as_processed.csv"):
    output_file = PROCESSED_DIR / filename
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ml_data.to_csv(output_file)

    print(f"Saved processed data to: {output_file}")
    print(f"Shape: {ml_data.shape}")

# MAIN PROGRAM

def run_pipeline():

    lines = load_geo_series_matrix("GSE73754_series_matrix.txt")

    sample_info = extract_sample_info(lines)
    expression_df = build_expression_matrix(lines)

    expression_df = preprocess_expression(expression_df)

    ml_data = build_ml_dataset(expression_df, sample_info)

    run_pca_lda(ml_data)

    save_dataset(ml_data)

    return ml_data
if __name__ == "__main__":
    df = run_pipeline()
    print(df.head())