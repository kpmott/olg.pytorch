#!/home/kpmott/Git/olg.pytorch/pyt/bin/python3

from packages import *
from params import *
from nn import *

# model.to("cuda")
data = CustDataSet(pretrain=True)
train_loader = DataLoader(data,batch_size=100,generator=torch.Generator(device="cuda"),shuffle=True)
trainer = pl.Trainer(max_epochs=500, accelerator="gpu",logger=False,enable_checkpointing=False)
trainer.fit(model=model,train_dataloaders=train_loader)

for thyme in tqdm(range(100)):
    data = CustDataSet()
    train_loader = DataLoader(data,batch_size=32,generator=torch.Generator(device="cuda"),shuffle=True)
    trainer = pl.Trainer(max_epochs=1, accelerator="gpu",logger=False,enable_checkpointing=False,enable_model_summary=False)
    trainer.fit(model=model,train_dataloaders=train_loader)

def plots():
    #Inspect output
    data = CustDataSet(cpu=True,sim=True)
    Σ, Y = np.array(data[:])[0].cuda(), np.array(data[:])[1].cuda()

    τ = len(Y)

    #here's where to do Euler/MCC loss
    E = Y[...,equity]
    B = Y[...,bond]
    P = Y[...,price]
    Q = Y[...,ir]

    #BC accounting: Consumption
    E_ = torch.nn.functional.pad(Σ[...,equity],(1,0))
    B_ = torch.nn.functional.pad(Σ[...,bond],(1,0))
    Ω_ = Σ[...,endow]
    Δ_ = Σ[...,div]
    Chat = Ω_ + (P+Δ_)*E_ + B_ - P*torch.nn.functional.pad(E,(0,1)) - Q*torch.nn.functional.pad(B,(0,1))
    ϵc = 1e-4
    C = torch.maximum(Chat,ϵc*(Chat*0+1))
    cpen = -torch.sum(torch.less(Chat,0)*Chat/ϵc)

    #Forecasting
    endog = torch.concat([E,B],-1)[None].repeat(S,1,1)
    exog = torch.tensor([[*[wvec[s]],*ωvec[s], *[δvec[s]]] for s in range(S)])[:,None,:].repeat(1,τ,1)
    Σf = torch.concat([endog,exog],-1).float()
    Yf = model(Σf)
    Ef = Yf[...,equity]
    Bf = Yf[...,bond]
    Pf = Yf[...,price]
    Qf = Yf[...,ir]
    Ef_ = torch.nn.functional.pad(E,(1,0))[None].repeat(S,1,1)
    Bf_ = torch.nn.functional.pad(B,(1,0))[None].repeat(S,1,1)
    Ωf_ = Σf[...,endow]
    Δf_ = Σf[...,div]
    Cf = Ωf_ + (Pf+Δf_)*Ef_ + Bf_ - Pf*torch.nn.functional.pad(Ef,(0,1,0,0,0,0)) - Qf*torch.nn.functional.pad(Bf,(0,1,0,0,0,0))

    #Euler Errors
    eqEuler = torch.sum(torch.abs(upinv(torch.tensordot(β*up(Cf[...,1:])/up(C[...,:-1])*(Pf+Δf_)/P,torch.tensor(probs),dims=([0],[0])))-1),-1)
    bondEuler = torch.sum(torch.abs(upinv(torch.tensordot(β*up(Cf[...,1:])/up(C[...,:-1])/Q,torch.tensor(probs),dims=([0],[0])))-1),-1)

    #Market Clearing Errors
    equityMCC = torch.abs(equitysupply-torch.sum(E,-1))
    bondMCC = torch.abs(bondsupply-torch.sum(B,-1))

    #per-period loss
    loss_vec = eqEuler + bondEuler + equityMCC + bondMCC + cpen

    #mse loss val
    lossfun = torch.nn.MSELoss()
    lossfun(loss_vec,loss_vec*0)

    Δ = Σ[:,div]
    eqRet = ((P[1:] + Δ[1:])/P[:-1])**(L/60) - 1
    bondRet = (1/Q[1:])**(L/60) - 1
    exRet = eqRet - bondRet

    plt.plot(C[-100:].cpu());plt.savefig('cons.png');plt.clf()
    plt.plot(exRet[-100:].cpu());plt.savefig('exRet.png');plt.clf()

    return C

C = plots()