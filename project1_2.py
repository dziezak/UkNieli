from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sktime.classification.dictionary_based import WEASEL
import pandas as pd
import numpy as np

# Wczytanie danych
df = pd.read_csv("tox21.csv")
target = "NR-AR"
df = df.dropna(subset=[target])

# Tokenizacja SMILES jako lista znaków
def tokenize_smiles(smiles):
    return pd.Series(list(smiles))  # np. "CCO" → ["C", "C", "O"]

# Tworzymy nested DataFrame
X_nested = pd.DataFrame([tokenize_smiles(s) for s in df["smiles"]])
X_nested = X_nested.apply(lambda row: row.dropna().reset_index(drop=True), axis=1)
X_nested = pd.DataFrame({"smiles": X_nested["smiles"]})  # kolumna z sekwencjami

y = df[target].values.astype(int)

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X_nested, y, test_size=0.2, random_state=42)

# Klasyfikator WEASEL z ograniczeniem liczby cech
clf = WEASEL(
    random_state=42,
    feature_selection="chi2",
    alphabet_size=2,
    bigrams=False,
    min_window=1
)

# Trening
clf.fit(X_train, y_train)

# Ewaluacja
y_pred = clf.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, y_pred)

print(f"ROC-AUC (WEASEL): {roc:.3f}")
