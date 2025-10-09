from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sktime.classification.dictionary_based import WEASEL
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.datatypes._panel._convert import from_2d_array_to_nested
import pandas as pd
import numpy as np

# Wczytanie danych
df = pd.read_csv("tox21.csv")
target = "NR-AR"
df = df.dropna(subset=[target])

# Tokenizacja SMILES jako stringi
def tokenize_smiles(smiles):
    return " ".join(list(smiles))  # prosta tokenizacja znak po znaku

df["smiles_tokenized"] = df["smiles"].apply(tokenize_smiles)

# Przekształcenie do formatu nested DataFrame (wymagane przez sktime)
X_text = df["smiles_tokenized"].values
y = df[target].values.astype(int)

# Zamiana na nested format
X_nested = from_2d_array_to_nested(np.array([[s] for s in X_text]))

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X_nested, y, test_size=0.2, random_state=42)

# Klasyfikator WEASEL z ograniczeniem liczby cech
clf = WEASEL(
    random_state=42,
    feature_selection="chi2",  # ogranicza liczbę cech
    alphabet_size=2,           # mniej symboli = mniejsza macierz
    bigrams=False              # wyłączenie bigramów
)

# Trening
clf.fit(X_train, y_train)

# Ewaluacja
y_pred = clf.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, y_pred)

print(f"ROC-AUC (WEASEL): {roc:.3f}")