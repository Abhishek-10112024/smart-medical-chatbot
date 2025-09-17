import pandas as pd

# ----------------------------
# Step 1: Load CSV
# ----------------------------
csv_path = "data/medical_qa.csv"  # change if your CSV name is different
df = pd.read_csv(csv_path)

# ----------------------------
# Step 2: Optional - take a small sample for prototyping
# ----------------------------
sample_df = df.head(20)  # first 20 rows for testing
# Use full df later: df = df

# ----------------------------
# Step 3: Clean text columns
# ----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    # remove extra spaces, newlines
    text = text.replace("\n", " ").strip()
    return text

sample_df["Patient"] = sample_df["Patient"].apply(clean_text)
sample_df["Doctor"] = sample_df["Doctor"].apply(clean_text)

# ----------------------------
# Step 4: Create combined Q/A column for embedding
# ----------------------------
sample_df["combined_text"] = sample_df["Patient"] + " [SEP] " + sample_df["Doctor"]

# ----------------------------
# Step 5: Save preprocessed CSV
# ----------------------------
preprocessed_path = "data/preprocessed_sample.csv"
sample_df.to_csv(preprocessed_path, index=False)
print(f"Saved preprocessed sample CSV to {preprocessed_path}")