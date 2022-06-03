#!/home/kpmott/Git/olg.pytorch/pyt/bin/python3

from packages import *
from params import *
from nn import *

model.to("cuda")
data = CustDataSet(pretrain=True)
train_loader = DataLoader(data,batch_size=32,generator=torch.Generator(device="cuda"),shuffle=True)
trainer = pl.Trainer(max_epochs=50, accelerator="gpu", )
trainer.fit(model=model,train_dataloaders=train_loader)

for thyme in tqdm(range(100)):
    data = CustDataSet()
    train_loader = DataLoader(data,batch_size=32,generator=torch.Generator(device="cuda"),shuffle=True)
    trainer = pl.Trainer(max_epochs=1, accelerator="gpu")
    trainer.fit(model=model,train_dataloaders=train_loader)