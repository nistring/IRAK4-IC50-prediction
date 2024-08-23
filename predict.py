import pandas as pd
import numpy as np
import torch
from lightning import pytorch as pl
from pathlib import Path

from chemprop import data, featurizers, models

from utils import pIC50_to_IC50

chemprop_dir = Path.cwd()
checkpoint_path = chemprop_dir / "checkpoints" / "epoch=19-step=500-v1.ckpt" # path to the checkpoint file.

mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)

test_path = chemprop_dir / "data" / "test.csv"
smiles_column = 'Smiles'

df_test = pd.read_csv(test_path)
smis = df_test[smiles_column]
test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in smis]

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
test_loader = data.build_dataloader(test_dset, shuffle=False)

with torch.inference_mode():
    trainer = pl.Trainer(
        logger=None,
        enable_progress_bar=True,
        accelerator="cpu",
        devices=1
    )
    test_preds = trainer.predict(mpnn, test_loader)

test_preds = np.concatenate(test_preds, axis=0)
test_preds = pIC50_to_IC50(test_preds)
df_test['IC50_nM'] = test_preds
df_test = df_test.drop("Smiles", axis=1)
df_test.to_csv("pred/test_submit.csv", index=False)