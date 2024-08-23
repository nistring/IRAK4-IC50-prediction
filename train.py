import pandas as pd
from pathlib import Path

from lightning import pytorch as pl

from chemprop import data, featurizers, models, nn
from utils import ScoreMetric
from multiprocessing import cpu_count

chemprop_dir = Path.cwd()
input_path = chemprop_dir / "data" / "train.csv" # path to your data .csv file
num_workers = cpu_count() # number of workers for dataloader. 0 means using main process for data loading
smiles_column = 'Smiles' # name of the column containing SMILES strings
target_columns = ['pIC50'] # list of names of the columns containing targets

df_input = pd.read_csv(input_path)
smis = df_input.loc[:, smiles_column].values
ys = df_input.loc[:, target_columns].values
all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

mols = [d.mol for d in all_data]  # RDkit Mol objects are use for structure based splits
train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1))
train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_dset = data.MoleculeDataset(train_data, featurizer)
scaler = train_dset.normalize_targets()

val_dset = data.MoleculeDataset(val_data, featurizer)
val_dset.normalize_targets(scaler)

test_dset = data.MoleculeDataset(test_data, featurizer)

train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)

mp = nn.BondMessagePassing()
agg = nn.MeanAggregation()
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
ffn = nn.RegressionFFN(output_transform=output_transform)
batch_norm = True

metric_list = [nn.metrics.RMSEMetric(), nn.metrics.MAEMetric(), ScoreMetric()]
mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=20, # number of epochs to train for
)

trainer.fit(mpnn, train_loader, val_loader)
results = trainer.test(mpnn, test_loader)
