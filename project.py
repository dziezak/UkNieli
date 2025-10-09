from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

df = pd.read_csv("tox21.csv")

target = "NR-AR"  
df = df.dropna(subset=[target])  # usuwamy brakujące etykiety

# SMILES -> fingerprint
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return np.zeros(1024)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))

# SMILES -> wektory
X = np.array([smiles_to_fp(s) for s in df["smiles"]])
y = df[target].values.astype(int)

# Dziele dane do treningu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model baseline – Random Forest ( czy moge korzstać z pythonowych Lasów czy mam pisać własne? )
clf = RandomForestClassifier(n_estimators=200, random_state=42) # randomowe wartości polecone z internetu dla tej bazy
# tutaj klasyfikatory stringowe!!! 
# sktime. Tutaj bęzie duża macierz. Weasel ???
# Weasel wygeneruje za dużą macierz i co z tym zrobić? 
clf.fit(X_train, y_train)

# Ewaluacja
y_pred = clf.predict_proba(X_test)[:, 1]  # prawdopodobieństwo klasy "toksyczna"
roc = roc_auc_score(y_test, y_pred)

print(f"ROC-AUC (baseline Random Forest): {roc:.3f}")
