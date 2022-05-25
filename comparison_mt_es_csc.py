#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 17:12:24 2019 

@author: elemhunt
"""
import numpy as np
import matplotlib.pyplot as plt
import math
#%% Model Functions

# Dilute Eshelby (ES)
def DilES(f,vm,vi,Em, Ei):
    import numpy as np
    # __M &&__I Values
    muM = Em/(2*(1+vm))
    muI = Ei/(2*(1+vi))
    kM = Em/(3*(1-2*vm))
    kI = Ei/(3*(1-2*vi))

    # __P Values
    muP = ((5/3)*muM*(3*kM + 4*muM)/(kM + 2*muM))/2
    kP = (3*kM + 4*muM)/3

    # Inital setting of arrays to be modified to store
    kes = np.zeros(len(f))
    mues = np.zeros(len(f))
    ves = np.zeros(len(f))
    ees = np.zeros(len(f))

    #Calculations
    for i in range(len(f)):
    
        kes[i] = kM +(f[i]*(kI - kM))/(1.+(kI - kM)/kP)
        mues[i] = muM +(f[i]*(muI - muM))/(1.+((muI - muM)/muP))
        ves[i] = (3.*kes[i]-2.*mues[i])/(6.*kes[i]+2.*mues[i])
        ees[i] = (2.*mues[i]*(1.+ves[i]))#*(1E-3)
    return kes, mues, ves, ees

# Mori-Tanaka (MT)
def MorT(f,vm,vi,Em, Ei):
    import numpy as np
    # __M &&__I Values
    muM = Em/(2*(1+vm))
    muI = Ei/(2*(1+vi))
    kM = Em/(3*(1-2*vm))
    kI = Ei/(3*(1-2*vi))

    # __P Values
    muP = ((5/3)*muM*(3*kM + 4*muM)/(kM + 2*muM))/2
    kP = (3*kM + 4*muM)/3
    
    # Inital setting of arrays to be modified to store
    kmt = np.zeros(len(f))
    mumt = np.zeros(len(f))
    vmt = np.zeros(len(f))
    emt = np.zeros(len(f))
    
    #Calculations
    for j in range(len(f)):
    
        kmt[j] = kM +(f[j]*(kI - kM))/(1.+(1-f[j])*(kI - kM)/kP)
        mumt[j] = muM +(f[j]*(muI - muM))/(1.+(1-f[j])*((muI - muM)/muP))
        vmt[j] = (3.*kmt[j]-2.*mumt[j])/(6.*kmt[j]+2.*mumt[j])
        emt[j] = (2.*mumt[j]*(1.+vmt[j]))#*(1E-3)
    return kmt, mumt, vmt, emt

# Classical Self Consistent (CSC)
def ClaSC(f,vm,vi,Em, Ei):
    import numpy as np
    tol = 1E-7
    
    # __M &&__I Values
    muM = Em/(2*(1+vm))
    muI = Ei/(2*(1+vi))
    kM = Em/(3*(1-2*vm))
    kI = Ei/(3*(1-2*vi))
    
    # Inital setting of arrays to be modified to store
    kcsc = np.zeros(len(f))
    mucsc = np.zeros(len(f))
    vcsc = np.zeros(len(f))
    ecsc = np.zeros(len(f))
    #Calculations
    for l in range(len(f)):
        muold = 0
        kold = 0
        mu = muM
        k = kM
        dif =1
        while(dif >= tol ):
            muP=0.5*(5/3)*mu*(3*kM+4*mu)/(kM+2*mu)
            mu = muM + (f[l]*(muI-muM))/(1+((muI-mu)/muP))
            kP = (3*k+4*mu)/3
            k = kM +(f[l]*(kI-kM))/(1+((kI-k)/kP))
            dif = math.sqrt((k-kold)**2 + (mu-muold)**2) 
            kold = k
            muold = mu
        mucsc[l] = mu
        kcsc[l] = k
        vcsc[l] = (3.*kcsc[l]-2.*mucsc[l])/(6.*kcsc[l]+2.*mucsc[l])
        ecsc[l] = (2.*mucsc[l]*(1.+vcsc[l]))
    return kcsc, mucsc, vcsc, ecsc
#%% Comparative Study Constants
# Volume Fraction
f = np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.30])
#%% Spherical Voids in Elastic Matrix

Em = 1.0   
vm = 0.25
Ei = 0.0 
vi = 0.0

# K lists
SV_ES_k =[]
SV_MT_k =[]
SV_CSC_k =[]

# MU lists
SV_ES_mu =[]
SV_MT_mu =[]
SV_CSC_mu =[]

# NU lists
SV_ES_ves = []
SV_MT_ves = []
SV_CSC_ves = []

# E List
SV_ES_ees =[]
SV_MT_ees =[]
SV_CSC_ees =[]


# Calculations
SV_ES_k, SV_ES_mu, SV_ES_ves, SV_ES_ees = DilES(f,vm,vi,Em, Ei)

SV_MT_k, SV_MT_mu, SV_MT_ves, SV_MT_ees = MorT(f,vm,vi,Em, Ei)

SV_CSC_k, SV_CSC_mu, SV_CSC_ves, SV_CSC_ees = ClaSC(f,vm,vi,Em, Ei)

# Plot E
fig, ax = plt.subplots()
l1 = ax.plot(f, SV_ES_ees,'-', color = 'green')
l2 = ax.plot(f, SV_MT_ees,'-', color = 'red')
l3 = ax.plot(f, SV_CSC_ees,'-', color = 'blue')
ax.grid(True)
ax.set_ylabel('Elastic Modulus (E)')
ax.set_xlabel('fi - Volume Fraction')
ax.set_title('Spherical Voids in Elastic Matrix - Elastic Mod.')
labels = ['Eshelby Strain','Mori-Tanaka','CSC']
fig.legend([l1, l2, l3],labels=labels,loc="best",title="Legend")

# Plot NU
fig, ax2 = plt.subplots()
l5 = ax2.plot(f, SV_ES_ves,'-', color = 'green')
l6 = ax2.plot(f, SV_MT_ves,'-', color = 'red')
l7 = ax2.plot(f, SV_CSC_ves,'-', color = 'blue')
ax2.grid(True)
ax2.set_ylabel('Poisson Ratio (NU)')
ax2.set_xlabel('fi - Volume Fraction')
ax2.set_title('Spherical Voids in Elastic Matrix NU')
labels = ['Eshelby Strain','Mori-Tanaka','CSC']
fig.legend([l5, l6, l7],labels=labels, loc="best", title="Legend")

#%% Rigid Spheres in Elastic Matrix

Em = 1
vm = 0.25 
Ei = 81E15  
vi = 0.25

# K lists
RS_ES_k =[]
RS_MT_k =[]
RS_CSC_k =[]

# MU lists
RS_ES_mu =[]
RS_MT_mu =[]
RS_CSC_mu =[]

# NU lists
RS_ES_ves = []
RS_MT_ves = []
RS_CSC_ves = []

# E List
RS_ES_ees =[]
RS_MT_ees =[]
RS_CSC_ees =[]

# Calculations
RS_ES_k, RS_ES_mu, RS_ES_ves, RS_ES_ees = DilES(f,vm,vi,Em, Ei)

RS_MT_k, RS_MT_mu, RS_MT_ves, RS_MT_ees = MorT(f,vm,vi,Em, Ei)

RS_CSC_k, RS_CSC_mu, RS_CSC_ves, RS_CSC_ees = ClaSC(f,vm,vi,Em, Ei)

# Plot E
fig, ax3 = plt.subplots()
l1 = ax3.plot(f, RS_ES_ees,'-', color = 'green')
l2 = ax3.plot(f, RS_MT_ees,'-', color = 'red')
l3 = ax3.plot(f, RS_CSC_ees,'-', color = 'blue')
ax3.grid(True)
ax3.set_ylabel('Elastic Modulus (E)')
ax3.set_xlabel('fi - Volume Fraction')
ax3.set_title('Rigid Spheres in Elastic Matrix - Elastic Mod.')
labels = ['Eshelby Strain','Mori-Tanaka','CSC']
fig.legend([l1, l2, l3],labels=labels,loc="best",title="Legend")

# Plot NU
fig, ax4 = plt.subplots()
l5 = ax4.plot(f, RS_ES_ves,'-', color = 'green')
l6 = ax4.plot(f, RS_MT_ves,'-', color = 'red')
l7 = ax4.plot(f, RS_CSC_ves,'-', color = 'blue')

ax4.grid(True)
ax4.set_ylabel('Poisson Ratio (NU)')
ax4.set_xlabel('fi - Volume Fraction')
ax4.set_title('Rigid Spheres in Elastic Matrix - NU')
labels = ['Eshelby Strain','Mori-Tanaka','CSC','Raw Data']
fig.legend([l5, l6, l7],labels=labels, loc="best", title="Legend")

#%% Glass Elastic Spheres in Epoxy Matrix

Em = 3.0 
vm = 0.38
Ei = 70
vi = 0.20



# K lists
GS_ES_k =[]
GS_MT_k =[]
GS_CSC_k =[]

# MU lists
GS_ES_mu =[]
GS_MT_mu =[]
GS_CSC_mu =[]

# NU lists
GS_ES_ves = []
GS_MT_ves = []
GS_CSC_ves = []

# E List
GS_ES_ees =[]
GS_MT_ees =[]
GS_CSC_ees =[]


# Calculations
GS_ES_k, GS_ES_mu,GS_ES_ves, GS_ES_ees = DilES(f,vm,vi,Em, Ei)

GS_MT_k, GS_MT_mu,GS_MT_ves, GS_MT_ees = MorT(f,vm,vi,Em, Ei)

GS_CSC_k, GS_CSC_mu, GS_CSC_ves, GS_CSC_ees = ClaSC(f,vm,vi,Em, Ei)



# Plot E
fig, ax5 = plt.subplots()
l1 = ax5.plot(f, GS_ES_ees,'-', color = 'green')
l2 = ax5.plot(f, GS_MT_ees,'-', color = 'red')
l3 = ax5.plot(f, GS_CSC_ees,'-', color = 'blue')

ax5.grid(True)
ax5.set_ylabel('Elastic Modulus (E)')
ax5.set_xlabel('fi - Volume Fraction')
ax5.set_title('Glass Elastic Spheres in Epoxy Matrix - Elastic Mod.')
labels = ['Eshelby Strain','Mori-Tanaka','CSC']
fig.legend([l1, l2, l3],labels=labels,loc="best",title="Legend")

# Plot NU
fig, ax6 = plt.subplots()
l5 = ax6.plot(f, GS_ES_ves,'-', color = 'green')
l6 = ax6.plot(f, GS_MT_ves,'-', color = 'red')
l7 = ax6.plot(f, GS_CSC_ves,'-', color = 'blue')
ax6.grid(True)
ax6.set_ylabel('Poisson Ratio (NU)')
ax6.set_xlabel('fi - Volume Fraction')
ax6.set_title('Glass Elastic Spheres in Epoxy Matrix - NU')

labels = ['Eshelby Strain','Mori-Tanaka','CSC']
fig.legend([l5, l6, l7],labels=labels, loc="best", title="Legend")
