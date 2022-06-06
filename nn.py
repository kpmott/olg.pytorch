from params import *
from packages import *
import detSS

ebar,bbar,pbar,qbar,xbar,cbar = detSS.detSS_allocs()

sizes = [input,2048,2048,1024,output]

class custAct(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        nn_tanh = nn.Tanh()
        nn_sp = nn.Softplus()
        act1 = nn_tanh(x[...,equity])
        act2 = nn_tanh(x[...,bond])
        act3 = nn_sp(x[...,price])
        act4 = nn_sp(x[...,ir])
        #act5 = x[...,2*L:]
        return torch.concat([act1,act2,act3,act4],dim=-1) #,act5

class MODEL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=sizes[0],out_features=sizes[1]),nn.ReLU(),nn.Dropout(p=0.1),
            nn.Linear(in_features=sizes[1],out_features=sizes[2]),nn.ReLU(),nn.Dropout(p=0.1),
            nn.Linear(in_features=sizes[2],out_features=sizes[3]),nn.ReLU(),nn.Dropout(p=0.1),
            nn.Linear(in_features=sizes[3],out_features=sizes[4]),custAct()
        )
        self.mse = nn.MSELoss()
        #self.pretrain = pretrain

    def forward(self,x):
        return self.model(x)
    
    def training_step(self,batch,batch_idx):
        x, y = batch
        y_pred = self.model(x)
        τ = len(y_pred)

        #here's where to do Euler/MCC loss
        E = y_pred[...,equity]
        B = y_pred[...,bond]
        P = y_pred[...,price]
        Q = y_pred[...,ir]

        #BC accounting: Consumption
        E_ = torch.nn.functional.pad(x[...,equity],(1,0))
        B_ = torch.nn.functional.pad(x[...,bond],(1,0))
        Ω_ = x[...,endow]
        Δ_ = x[...,div]
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

        loss_vec = eqEuler + bondEuler + equityMCC + bondMCC

        #Total Loss
        pretrain = (torch.max(y) > 0.)
        if pretrain: 
            loss = self.mse(y_pred,y)
        else:
            loss = self.mse(loss_vec,loss_vec*0)
        
        return loss

    def predict_step(self,batch,batch_idx):
        X_batch, Y_batch = batch
        preds = self.model(X_batch.float())
        return preds
    
    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=1e-7)

model = MODEL().to("cuda")

class CustDataSet(Dataset):
    def __init__(self,pretrain=False,cpu=False,sim=False):
        model.to("cuda")
        model.eval()
        Σ = torch.zeros(T,input)
        Y = torch.zeros(T,output)
        Σ[0] = torch.tensor([*ebar,*bbar,*[wbar],*ω_scalar,*[δ_scalar]])
        Y[0] = model(Σ[0])
        shist, whist, Ωhist, Δhist, Ω, Δ = SHOCKS()
        for t in range(1,T):
            Σ[t] = torch.tensor([[*Y[t-1,equity],*Y[t-1,bond],*[whist[t]],*Ωhist[t],*[Δhist[t]]]])
            Y[t] = model(Σ[t])
        
        Σ, Y = Σ[time], Y[time]

        if pretrain:
            self.Y = torch.ones(train,output)*torch.tensor([*ebar,*bbar,*[pbar],*[qbar]]).float()
        elif sim:
            self.Y = Y.detach().clone()
        else:
            self.Y = torch.zeros(train,output).float()

        if cpu:
            self.Σ = Σ.detach().clone().cpu()
            self.Y = self.Y.cpu()
        else: 
            self.Σ = Σ.detach().clone()

        model.train()

    def __len__(self):
        return len(self.Y)
    
    #HERE USE NN TO START FROM detSS AND THEN BUILD DATA
    def __getitem__(self,idx):
        return self.Σ[idx], self.Y[idx]