import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer

from torch.optim import Adam
import pandas as pd

class PinsoroBoredom(LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer_1 = nn.Linear(17, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32,2)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        return x

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):

        x, y = batch['in'], batch['classes']
        out = self(x)
        y_pred = F.softmax(out, dim=1)

        return torch.argmax(y_pred, dim=1) == 0

    def training_step(self, batch, batch_idx):
        x, y = batch['in'], batch['classes']
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('cross_entropy_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['in'], batch['classes']
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log('validation_cross_entropy_loss', loss)
        return loss
 

    def test_step(self, batch, batch_idx):
        x, y = batch['in'], batch['classes']
        embedding = self(x)
        loss = F.cross_entropy(embedding, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

n = 100

class Pinsoro(Dataset):

    # 1: socially active, engaged; 2: dis-engaged
    CLASSES = {'assertive': 1,
               'adversarial': 1, 
               'passive': 0, 
               'prosocial': 1,
               'frustrated': 0}

    def __init__(self, csv_file, transform=None) -> None:
        super().__init__()

        print("Loading dataset... this might take some time!")
        df = pd.read_csv(csv_file)
        l1 = len(df)
        df.drop(df[df.purple_child_task_engagement.isna() | 
                    df.purple_child_au01.isna()].index, inplace=True)

        print("%s samples available in this chunk (-%s)" % (len(df), l1-len(df)))

        self.pinsoro = df
        print("Dataset loaded.")

    def __len__(self):
        return len(self.pinsoro)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            # if idx is not a scalar, the code below (eg 'split' string) will likely break!!
            import pdb;pdb.set_trace()
            idx = idx.tolist()

        aus = list(self.pinsoro.iloc[idx, 195:213].dropna()) # purple; 404-422 yellow
        task_engagement = self.pinsoro.iloc[idx, 443] #purple; 446 yellow
        social_engagement = self.pinsoro.iloc[idx, 444] # purple; 447 yellow
        social_attitude = self.pinsoro.iloc[idx, 445] # purple; 448 yellow

        # if 2+ annotators disagreed on the annotations, annotations are concatenated with '+'. In that case, simply keep the first one.
        social_attitude = social_attitude.split('+')[0]

        input_features = np.array(aus).astype(np.single)
        output_classes = np.array(Pinsoro.CLASSES[social_attitude])

        sample ={'in': input_features, 'classes':output_classes}

        return sample



if __name__ == "__main__":

    pinsoro = Pinsoro('pinsoro.csv')
    pinsoro_train, pinsoro_val, pinsoro_test = random_split(pinsoro, [int(len(pinsoro) * 0.7), int(len(pinsoro) * 0.1), int(len(pinsoro) * 0.2)])

    pinsoro_train = DataLoader(pinsoro_train, batch_size=256, shuffle=True, num_workers=8)
    pinsoro_val = DataLoader(pinsoro_val, batch_size=256, shuffle=True, num_workers=8)
    pinsoro_test = DataLoader(pinsoro_test, batch_size=256, shuffle=True, num_workers=8)

    model = PinsoroBoredom()
    trainer = Trainer(max_epochs=100)
    #trainer = Trainer(resume_from_checkpoint="lightning_logs/version_0/checkpoints/epoch=263-step=161043.ckpt")
    trainer.fit(model, pinsoro_train, pinsoro_val)

    result = trainer.test(model, pinsoro_test)
    print(result)

    #trainer.predict(model, pinsoro_test)


