from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

# 1. Wczytaj dane
df = pd.read_csv("tox21.csv")  # zakładam, że plik już masz

# 2. Wybierz jedną etykietę do klasyfikacji, np. NR-AR
target = "NR-AR"  
df = df.dropna(subset=[target])  # usuń brakujące etykiety

# 3. Funkcja: SMILES -> fingerprint
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # obsłuż błędne SMILES
        return np.zeros(1024)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))

# 4. Zamiana wszystkich SMILES na wektory
X = np.array([smiles_to_fp(s) for s in df["smiles"]])
y = df[target].values.astype(int)

# 5. Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model baseline – Random Forest
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# 7. Ewaluacja
y_pred = clf.predict_proba(X_test)[:, 1]  # prawdopodobieństwo klasy "toksyczna"
roc = roc_auc_score(y_test, y_pred)

print(f"ROC-AUC (baseline Random Forest): {roc:.3f}")
