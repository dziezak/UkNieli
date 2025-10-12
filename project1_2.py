from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sktime.classification.dictionary_based import WEASEL
import pandas as pd
import numpy as np

# Wczytanie danych
df = pd.read_csv("tox21.csv")
target = "NR-AR"
df = df.dropna(subset=[target])
df = df[df["smiles"].apply(lambda s: len(s) >= 6)]

# Tworzymy słownik znaków SMILES → liczby
unique_chars = sorted(set("".join(df["smiles"])))
char_to_int = {char: idx + 1 for idx, char in enumerate(unique_chars)}  # +1 żeby uniknąć zera

# Funkcja kodująca SMILES jako sekwencję liczb
def encode_smiles(smiles, length=12):
    seq = [char_to_int.get(c, 0) for c in smiles]  # 0 dla nieznanych
    if len(seq) < length:
        seq += [0] * (length - len(seq))  # padding
    else:
        seq = seq[:length]  # obcięcie
    return pd.Series(seq)

# Tworzymy nested DataFrame z sekwencjami numerycznymi
max_len = 12
X_nested = pd.DataFrame({"smiles": [encode_smiles(s, max_len) for s in df["smiles"]]})
y = df[target].values.astype(int)

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X_nested, y, test_size=0.2, random_state=42)

# Klasyfikator WEASEL
clf = WEASEL(
    random_state=42,
    feature_selection="chi2",
    alphabet_size=4,
    bigrams=False
)

# Trening
clf.fit(X_train, y_train)

# Ewaluacja
y_pred = clf.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, y_pred)

print(f"ROC-AUC (WEASEL): {roc:.3f}")
