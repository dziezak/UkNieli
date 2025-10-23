# =============================================
# PROJEKT: Klasyfikacja toksyczności – zbiór Tox21
# =============================================

# 🔧 Importy
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm

# =============================================
# 1. Wczytanie danych
# =============================================

# Zmieniasz tę ścieżkę na własną, np. "tox21.csv"
# (Plik powinien zawierać kolumny: smiles, mol_id, NR-AR, NR-AR-LBD, ... SR-p53)
data = pd.read_csv("tox21.csv")

print("Liczba rekordów:", len(data))
print("Kolumny w zbiorze:", data.columns.tolist())

# =============================================
# 2. Sprawdzenie poprawności SMILES
# =============================================

def is_valid_smiles(smiles):
    """Zwraca True, jeśli SMILES da się zamienić na cząsteczkę."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

data["valid_smiles"] = data["smiles"].apply(is_valid_smiles)
data = data[data["valid_smiles"] == True].copy()
print("Poprawne SMILES:", len(data))

# =============================================
# 3. Wyciągnięcie cech z SMILES
# =============================================

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "LogP": Descriptors.MolLogP(mol),
    }

features = []
for s in tqdm(data["smiles"], desc="Ekstrakcja cech"):
    f = smiles_to_features(s)
    features.append(f)

features_df = pd.DataFrame(features)
print("Cechy molekularne:", features_df.shape)

# Dodajemy cechy do oryginalnych danych
data = pd.concat([data.reset_index(drop=True), features_df], axis=1)

# =============================================
# 4. Przygotowanie X (cechy) i Y (etykiety toksyczności)
# =============================================

target_columns = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

# X = cechy wejściowe
X = data[["MolWt", "NumHDonors", "NumHAcceptors", "NumAromaticRings",
          "TPSA", "NumRotatableBonds", "LogP"]].values

# Y = etykiety toksyczności
Y = data[target_columns]

# =============================================
# 5. Trening – przykład dla jednej etykiety (NR-AR)
# =============================================

label = "NR-AR"

# Usuwamy NaN (brak danych)
df = data.dropna(subset=[label])
X_label = df[["MolWt", "NumHDonors", "NumHAcceptors",
              "NumAromaticRings", "TPSA", "NumRotatableBonds", "LogP"]]
y_label = df[label].astype(int)

# Podział na train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_label, y_label, test_size=0.2, random_state=42, stratify=y_label
)

# Model
model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# =============================================
# 6. Ewaluacja
# =============================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)

print(f"\n=== Wyniki dla {label} ===")
print(f"Accuracy: {acc:.3f}")
print(f"ROC AUC:  {auc:.3f}")
print(f"F1 score: {f1:.3f}")

# =============================================
# 7. (Opcjonalnie) Trenowanie dla wszystkich etykiet
# =============================================

results = []
for col in target_columns:
    df = data.dropna(subset=[col])
    if df[col].nunique() < 2:  # pomiń kolumny bez obu klas
        continue

    X_label = df[["MolWt", "NumHDonors", "NumHAcceptors",
                  "NumAromaticRings", "TPSA", "NumRotatableBonds", "LogP"]]
    y_label = df[col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_label, y_label, test_size=0.2, random_state=42, stratify=y_label
    )

    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "label": col,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print("\n=== Podsumowanie dla wszystkich testów toksyczności ===")
print(results_df.sort_values("auc", ascending=False))
