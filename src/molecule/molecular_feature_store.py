from feast import Entity, FeatureView, Field
from feast import FeatureStore
from feast.types import Float32
from feast.file_source import FileSource


# rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd


PATH = ""

molecular_source = FileSource(
    path=PATH,
    timestamp_field = "event_timestamp"
)


# Entity
molecule = Entity(name = "molecule_id", join_keys = ["molecule_id"])

# Feature engineering function

def compute_rdkit_features(df: pd.DataFrame) -> pd.DataFrame:
    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
    df["mol_weight"] = df["mol"].apply(Descriptors.MolWt)
    df["logp"] = df["mol"].apply(Descriptors.MolLogP)
    return df[["molecule_id", "mol_weight", "logp", "event_timestamp"]]




