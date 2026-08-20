"""Small historical and online retrieval example."""

from datetime import datetime, timezone

import pandas as pd
from feast import FeatureStore


store = FeatureStore(repo_path="feature_repo")
features = [
    "molecule_features_v1:molecular_weight",
    "molecule_features_v1:logp",
    "molecule_features_v1:tpsa",
]
entities = pd.DataFrame(
    {"identity_smiles": ["CCO", "c1ccccc1"], "event_timestamp": [datetime.now(timezone.utc)] * 2}
)

historical = store.get_historical_features(entity_df=entities, features=features).to_df()
print("Historical (point-in-time join):")
print(historical.to_string(index=False))

online = store.get_online_features(
    features=features,
    entity_rows=[{"identity_smiles": "CCO"}, {"identity_smiles": "c1ccccc1"}],
).to_df()
print("\nOnline:")
print(online.to_string(index=False))
