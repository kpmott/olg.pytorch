import os
os.chdir("/home/kpmott/Git/olg.pytorch")

from packages import *

#Lifespan 
L = 3

#working periods and retirement periods 
wp = int(L*2/3)
rp = L - wp

#Time discount rate
β = 1.#0.995**(60/L)

#Risk-aversion coeff
γ = 2.

#Stochastic elements
probs = [0.5, 0.5]
S = len(probs) 

divshare = .05

#total resources
wbar = 1

#share of total resources
ωGuess = norm.pdf(np.linspace(1,wp,wp),0.8*wp,L*0.25)
ωGuess = (1-divshare)*ωGuess/np.sum(ωGuess)

#share of total resources: 1/8 to dividend; the rest to endowment income
# ls = wbar*np.array([*[divshare], *ωGuess, *np.zeros(rp)])
ls = np.array([1/3, 1/4, 5/12, 0])


#shock perturbation vector
ζtrue = 0.05
wvec = np.array([wbar*(1 - ζtrue), wbar*(1+ + ζtrue)])     #total resources in each state
δvec = np.multiply(ls[0],wvec)/wbar          #dividend in each state
ωvec = [ls[1:]*w/wbar for w in wvec]         #endowment process

ω = torch.tensor(np.array(ωvec))
δ = torch.reshape(torch.tensor(δvec),(S,1))

#mean-center all shock-contingent values
δ_scalar = ls[0]
ω_scalar = ls[1:]

#net supply of assets: for later
equitysupply = 1
bondsupply = 0

#-----------------------------------------------------------------------------------------------------------------
#utility
def u(x):
    if γ == 1:
        return np.log(x)
    else:
        return (x**(1-γ))/(1-γ)

#utility derivative
def up(x):
    return x**-γ

#inverse of utility derivative
def upinv(x):
    return x**(1/γ)

#-----------------------------------------------------------------------------------------------------------------
#time and such for neurals 
T = 7500
burn = int(T/100)            #burn period: this is garbage
train = T - burn            #how many periods are "counting?"
time = slice(burn,T,1)      #period in which we care

def SHOCKS():
    #shocks
    shocks = range(S)
    
    #Shock history:
    shist = np.random.choice(shocks,T,probs)

    #History: endowments and dividends and total resources
    Ωhist = [ωvec[t] for t in shist]
    Δhist = δvec[shist]
    whist = wvec[shist]

    #convert to tensors now for easier operations later
    Ω = torch.tensor(np.array(Ωhist))
    Δ = torch.reshape(torch.tensor(Δhist),(T,1))

    return shist, whist, Ωhist, Δhist, Ω, Δ

#machine tolerance
ϵ = 1e-8
#-----------------------------------------------------------------------------------------------------------------
"""
input   = [(e_i^{t-1})_i,(b_i^{t-1})_i,w^t,(ω_i^t)_i,t]
output  = [((e_i^{t})_{i=1}^{L-1},(b_i^{t})_{i=1}^{L-1},p^t,q^t]   ∈ ℜ^{2L}
"""
#input/output dims
#        assets     + resources     + endowments    + div   + time
input = 2*(L-1)     + 1             + L             + 1     #+ 1

#        assets     + prices   + endowments    + div   + time
output = 2*(L-1)    + 2        #+ L             + 1     + 1
output_endog = 2*(L-1)+2

#slices to grab output 
equity =    slice(0     ,L-1    ,1)
bond =      slice(L-1   ,2*L-2  ,1)
price =     slice(2*L-2 ,2*L-1  ,1)
ir =        slice(2*L-1 ,2*L    ,1)

#slices to grab input
endow = slice(2*L-1,    3*L-1,  1)
div =   slice(3*L-1,    3*L,    1)