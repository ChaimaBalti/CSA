#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on FEV 13 2024

@author: Chaima Balti 2AMIndS
"""
#******************************************************************************
# Resolution des lois de conservation 1D : 
#     o Equation du transport.
#     o Equation de Burgers.
#------------------------------------------------------------------------------
# 
#    dt(u) + dx f(u) = 0;  x \in [x_min,x_max], t>0
#    u(x,0) = u0(x);   x \in [x_min,x_max]
#    u(x_min,t) = u(x_max,t),  t>0.
#
# par des schemas explicites a deux niveaux de temps et trois pas d'espace :
#
# u(n+1,j) = H_n(u(n,j-1),u(n,j),u(n,j+1)).
#
# h            : pas d'espace donne (x_max-x_min)/N
# alpha        : rapport nominal dt/dx
# x_min, x_max : intervalle de resolution spatial
# T            : Temps final,
# N            : nombre de points en x
# NT           : nombre d'iterations en temps
# u0(x)        : fonction donnee initiale
# uex(x,t)     : solution exacte si connue
#
# le pas de temps est determine par la regle suivante de type CFL
# generalise
#
# dt(n) = alpha h / sup_j | a(u(n,j)) |.
#
# ou a(u) = f'(u).
#
#******************************************************************************


# Load packages----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize_scalar

# Definitions des fonctions:---------------------------------------------------

def FluxTransport(u,nderiv):
    '''
     Flux associe a l'equation de Transport.
     Parameters
    --------------------------------------------------------------------------
     u      : valeur pour laquelle on veut calculer le flux (ou la derivee du flux)
     nderiv : vaut 0 si on souhaite calculer le flux ou 1 si
              l'on souhaite calculer la derivee du flux
    --------------------------------------------------------------------------


    Returns
    -------
    Flux associe a l'equation de Burgers et (ou) son gradient.

    '''
    
    c=0.8
    if nderiv==0 :
        return c*u
    else:
        return c*np.ones(np.shape(u))
    
#------------------------------------------------------------------------------
    
def FluxBurgers(u,nderiv):
    '''
    Flux associe a l'equation de Burgers.
    Parameters
    ---------------------------------------------------------------------------
    u      : valeur pour laquelle on veut calculer le flux (ou la derivee du flux)
    nderiv : vaut 0 si on souhaite calculer le flux ou 1 si
             l'on souhaite calculer la derivee du flux
    ---------------------------------------------------------------------------
    
    
    Returns
    ---------------------------------------------------------------------------
    Flux associe a l'equation de Burgers et (ou) son gradient.

    '''
    if nderiv==0 :
        return (0.5)*(np.power(u,2))
    else:
        return u

#------------------------------------------------------------------------------

def Creneau(x):
    '''
    Cette fonction Creneau définit une fonction créneau pour 
    l'utiliser comme une condition initiale
    
    Parameters
    ---------------------------------------------------------------------------
    x : vecteur des reels
    
    Returns
    ---------------------------------------------------------------------------
    y : vecteur des reels verifient y=creneau(x)

    '''

    N=np.size(x)
    y=np.zeros(N)
    mask1=((x>=-2)*(x<-1/2))+((x>1/2)*(x<=2))
    mask2=(x>=-1/2)*(x<1/2)
    y=1*mask2+(-0.5)*mask1 
    
    return y
 
#------------------------------------------------------------------------------
   
def gamma(t):
    '''
    Cette fonction définit la fonction gamma pour 
    le calcule de la solution entropique 
    
    Parameters
    ---------------------------------------------------------------------------
    t : reel (temps)
    
    Returns
    ---------------------------------------------------------------------------
    reels verifient gamma(t)

    '''
    return (3*np.sqrt(1+t)-1-t)*0.5

#------------------------------------------------------------------------------

def Rampe(x):
    '''
    Cette fonction Rampe définit une fonction rampe pour 
    l'utiliser comme une condition initiale
    
    Parameters
    ---------------------------------------------------------------------------
    x : vecteur des reels
    
    Returns
    ---------------------------------------------------------------------------
    y : vecteur des reels verifient y=Rampe(x)


    '''
    
    N=np.size(x)
    y=np.zeros(N)
    mask1=((x>=-2)*(x<-1/2))+((x>=1)*(x<=2))
    mask2=(x>=-1/2)*(x<1)
    y=x*mask2+(-0.5)*mask1 
    
    return y

#------------------------------------------------------------------------------    

def aetoile(U,V, flux):
    
    '''
    la fonction utiliser pour calculer la fonction g du schema de MurmanRoe.
    
    Parameters
    ---------------------------------------------------------------------------
    U      : 1er vecteur des valeurs pour laquelle on veut calculer aetoile
    V      : 2nd vecteur des valeurs pour laquelle on veut calculer aetoile
    flux   : le flux utilisé : (flux de Burgers ou flux de transport)
    ---------------------------------------------------------------------------
    
    
    Returns
    ---------------------------------------------------------------------------
    vecteur des valeurs de aetoile
    '''
    
    fU = flux(U, 0)
    fV = flux(V, 0)
    fprime= flux(U, 1)
    
    mask=(U==V)
    
    return np.sign(fprime)*mask+(1-mask)*(np.sign(fU-fV)*np.sign(U-V))

#------------------------------------------------------------------------------
  
def g(U,V, flux):
    '''
    la fonction utiliser pour calculer le schema de MurmanRoe.
    
    Parameters
    ---------------------------------------------------------------------------
    U      : 1er vecteur des valeurs pour laquelle on veut calculer g(U)
    V      : 2nd vecteur des valeurs pour laquelle on veut calculer g(V)
    flux   : le flux utilisé : (flux de Burgers ou flux de transport)
    ---------------------------------------------------------------------------
    
    
    Returns
    ---------------------------------------------------------------------------
    vecteur des valeurs de g
    '''
    a=aetoile(U,V, flux)
    mask1=(a==0)+(a==1)
    mask2=(a==-1)
    fU = flux(U, 0);
    fV = flux(V, 0);
    return mask1*fU+mask2*fV

#------------------------------------------------------------------------------

def gG(U,V, flux):
    '''
    la fonction utiliser pour calculer le schema de Godunov, elle calcule 
    le maximum ou le minimum d'une fonction f(w) sur l'intervalle [v, u] ou 
    [u, v] respectivement, selon la relation entre u et v.

    
    Parameters
    ---------------------------------------------------------------------------
    U      : 1er vecteur des valeurs pour laquelle on veut calculer gG(U)
    V      : 2nd vecteur des valeurs pour laquelle on veut calculer gG(V)
    flux   : le flux utilisé : (flux de Burgers ou flux de transport)
    ---------------------------------------------------------------------------
    
    
    Returns
    ---------------------------------------------------------------------------
    fW  : vecteur des valeurs maximale ou minimale de f(w) selon la relation 
    entre u et v.

    '''
    fW=np.zeros([np.size(U),1]);
    for i in range(np.size(U)):
        if U[i]<=V[i]:
            def fct1(W):
                return flux(W,0)
            fW[i]=(minimize_scalar(fct1 ,bounds=(U[i], V[i]), method='bounded').fun) 
        else :
            def fct2(W):
                return -1*flux(W,0)
            fW[i]=-1*(minimize_scalar(fct2 ,bounds=(V[i], U[i]), method='bounded').fun)  
    return fW

#------------------------------------------------------------------------------

def Decentre(U,V,W,dt_sur_h, flux):
    '''
    la fonction utiliser pour calculer le schema décentre.
    
    Parameters
    ---------------------------------------------------------------------------
    U       : 1er vecteur des valeurs  (Uj-1) à l'instant tn
    V       : 2nd vecteur des valeurs  (Uj)   à l'instant tn
    W       : 3em vecteur des valeurs  (Uj+1) à l'instant tn
    dt_sur_h: la pas du temps sur la pas de l'éspace
    flux    : le flux utilisé (flux de Burgers ou flux de transport)
    ---------------------------------------------------------------------------
    
    Returns
    ---------------------------------------------------------------------------
    H : vecteur des valeurs Uj à l'instant tn+1
    '''
    H=np.zeros([np.size(U),1]);
    
    fU = flux(U, 0);
    fV = flux(V, 0);
    fW = flux(W, 0);
    
    a=flux(V, 1);
    
    mask=(a>=0)

    H=mask*(V-dt_sur_h*(fV-fU))+(1-mask)*(V-dt_sur_h*(fW-fV))
    
    return H

#------------------------------------------------------------------------------

def MurmanRoe(U,V,W,dt_sur_h, flux):
    '''
    la fonction utiliser pour calculer le schema du MurmanRoe.
    
    Parameters
    ---------------------------------------------------------------------------
    U       : 1er vecteur des valeurs  (Uj-1) à l'instant tn
    V       : 2nd vecteur des valeurs  (Uj)   à l'instant tn
    W       : 3em vecteur des valeurs  (Uj+1) à l'instant tn
    dt_sur_h: la pas du temps sur la pas de l'éspace
    flux    : le flux utilisé (flux de Burgers ou flux de transport)
    ---------------------------------------------------------------------------
    
    Returns
    ---------------------------------------------------------------------------
    H : vecteur des valeurs Uj à l'instant tn+1
    '''
    H=np.zeros([np.size(U),1]);
    
    gVW = g(V,W, flux)
    gUV = g(U,V, flux)
    H=V-dt_sur_h*(gVW-gUV)
    return H

#------------------------------------------------------------------------------

def LaxFriedrichs(U,V,W,dt_sur_h, flux):
    '''
    la fonction utiliser pour calculer le schema du LaxFriedrichs.
    
    Parameters
    ---------------------------------------------------------------------------
    U       : 1er vecteur des valeurs  (Uj-1) à l'instant tn
    V       : 2nd vecteur des valeurs  (Uj)   à l'instant tn
    W       : 3em vecteur des valeurs  (Uj+1) à l'instant tn
    dt_sur_h: la pas du temps sur la pas de l'éspace
    flux    : le flux utilisé (flux de Burgers ou flux de transport)
    ---------------------------------------------------------------------------
    
    Returns
    ---------------------------------------------------------------------------
    H : vecteur des valeurs Uj à l'instant tn+1
    '''
    H=np.zeros([np.size(U),1]);

    fU = flux(U, 0)
    fW = flux(W, 0)

    H=0.5*((U+W)-dt_sur_h*(fW-fU))
    
    return H

#------------------------------------------------------------------------------

def Godunov(U,V,W,dt_sur_h, flux):
    '''
    la fonction utiliser pour calculer le schema de Godunov.
    
    Parameters
    ---------------------------------------------------------------------------
    U       : 1er vecteur des valeurs  (Uj-1) à l'instant tn
    V       : 2nd vecteur des valeurs  (Uj)   à l'instant tn
    W       : 3em vecteur des valeurs  (Uj+1) à l'instant tn
    dt_sur_h: la pas du temps sur la pas de l'éspace
    flux    : le flux utilisé (flux de Burgers ou flux de transport)
    ---------------------------------------------------------------------------
    
    Returns
    ---------------------------------------------------------------------------
    H : vecteur des valeurs Uj à l'instant tn+1
    '''
    
    H=np.zeros([np.size(U),1])
    
    fplus  = gG(V,W,flux)
    fminus = gG(U,V,flux)
    
    H=V-dt_sur_h*(fplus-fminus)
    
    return H

#------------------------------------------------------------------------------


# Paramètres du probleme-------------------------------------------------------

#Le flux qui determine la lois de conservation a resoudre  f(u)
#flux  = FluxTransport   # Flux pour l equation de transport.
flux = FluxBurgers       # Flux pour l equation de Burgers. 


#Condition intiale a utiliser
#ConditionInitiale = Rampe
ConditionInitiale = Creneau  


# Schema a utiliser (expression de H(u,v,w))
#Hscheme = Decentre          
#Hscheme = MurmanRoe 
#Hscheme = LaxFriedrichs 
Hscheme = Godunov 


# Discretisation spatiale
N = 200 

# Discretisation temporelle
alpha = 0.5

# Variable valant 1 si l'on veut afficher la solution exacte
# Attention, fonctionne pour :
#  -Tous les problemes de transports lineaire
#  -Burgers avec condition initiale rampe 
sol_exacte=1 


#------------------------------------------------------------------------------
# Le code qui suit ne doit pas etre modifier dans le cadre du TP
#------------------------------------------------------------------------------

# Domaine spatiale et temporel
xmin = -2 
xmax = 2 
Tmin = 0  
Tmax = 5

# Discretisation spatial
h = (xmax-xmin)/N 
x=np.linspace(xmin, xmax-h, N) 

# Initialisation du compteur d'itérations et du temps initial
n=0 
t=Tmin 

# Condition initiale
u=np.zeros([N,1])
u[:,0]= ConditionInitiale(x) 


# Estimation du nombre maximale d'iterations en temps
dt = alpha * h / np.max( np.abs( flux(u[:,0],1) ) ) 
NT = int(np.floor( (Tmax - Tmin) / dt )) 
tt=np.zeros([NT,1])
dtt=np.zeros([NT,1])
tt[0]=t
dtt[0]=dt
uu=np.zeros([NT,N])
uu[0,:]=u[:,0]

# Solution exacte
uexact=np.zeros([NT,N])
uexact[0,:]= np.copy(u[:,0])

# Estimation de la norme infinie de la solution
LinftyCondInit = np.max(np.abs(u))

# calcul du min et max pour regler les axes
umin = np.min(u[0,:])
umax = np.max(u[0,:])
du = np.abs(umax-umin)
umin = umin - 0.1*du
umax = umax + 0.1*du


#------------------------------------------------------------------------------
# boucle en temps
#------------------------------------------------------------------------------

U=np.zeros([N,1]) #j-1
V=np.zeros([N,1]) #j
W=np.zeros([N,1]) #j+1
n=0
while (tt[n] < Tmax): 
    
    # Calcul du dt adaptatif
    #       alpha * h
    # dt = -----------------
    #       sup | a( u(n,:) ) |

    dt = alpha * h / np.max( np.abs( flux(uu[n,:],1) ) ) 
    dtt[n]=dt
    tt[n+1] = tt[n]+dtt[n] 
    
    U[1:N,0]=uu[n,0:N-1]
    U[0,0]=uu[n,N-1]
    
    V[:,0] = uu[n,:]
    
    W[0:N-1,0]=uu[n,1:N]
    W[N-1,0]=uu[n,0]
    
    # on applique le schema u_j^{n+1} = H(U,V,W)  
    u = Hscheme(U, V, W, dtt[n]/h, flux)
    
    uu[n+1,:]=u[:,0];  #Valeur approchée a t=n+1
    
    
    # calcul de la solution exacte.
    if (flux==FluxTransport):
        Xmod=xmin+np.mod(x-xmin-flux(0,1)*tt[n+1],(xmax-xmin))
        uexact[n+1,:] =  ConditionInitiale(Xmod)
    elif (flux==FluxBurgers):
        uexact[0,:]=ConditionInitiale(x)
        if (ConditionInitiale==Rampe):
            maskE=((x>=-2)*(x<=(-0.5)*(1+tt[n+1])))+((x>gamma(tt[n+1]))*(x<=2))
            maskE2=(x>=(-0.5)*(1+tt[n+1]))*(x<gamma(tt[n+1]))
            uexact[n+1,:] =  -0.5*maskE + (x/(1+tt[n+1]))*maskE2
        if (ConditionInitiale==Creneau):   
            if (tt[n+1]<=4/3):
              mask_1=(x>=-2)&(x<-0.5*(1+tt[n+1]))
              mask_2=(x>=-0.5*(1+tt[n+1]))&(x<tt[n+1]-0.5)
              mask_3=(x>tt[n+1]-0.5)&(x<0.25*tt[n+1]+0.5)
              mask_4=(x>=0.25*tt[n+1]+0.5)&(x<=2)
              ind_1=np.where(mask_1)
              uexact[n+1,ind_1]=-0.5
              ind_2=np.where(mask_2)
              uexact[n+1,ind_2]=(x[ind_2]+0.5)/tt[n+1]
              ind_3=np.where(mask_3)
              uexact[n+1,ind_3]=1
              ind_4=np.where(mask_4)
              uexact[n+1,ind_4]=-0.5
            else:
              gamma= np.sqrt(3*tt[n+1])-0.5*tt[n+1]-0.5
              mask_5=(x>=-2)&(x<=-0.5*(1+tt[n+1]))
              mask_6=(x>=-0.5*(1+tt[n+1]))&(x<=gamma)
              mask_7=(x>gamma)&(x<=2)
              ind_5=np.where(mask_5)
              uexact[n+1,ind_5]=-0.5
              ind_6=np.where(mask_6)
              uexact[n+1,ind_6]=(x[ind_6]+0.5)/tt[n+1]
              ind_7=np.where(mask_7)
              uexact[n+1,ind_7]=-0.5       
    
    n=n+1
    if n>=NT-1:
        break
NT=n
plt.figure(1) 
n=0

#Titre de la figure:-----------------------------------------------------------

if Hscheme==Decentre :
    titre='Schema décentré'
elif Hscheme==LaxFriedrichs :
    titre='Schema de LaxFriedrichs'
elif Hscheme==MurmanRoe :
    titre='Schema de MurmanRoe'
elif Hscheme==Godunov :
    titre='Schema de Godunov'
    
if flux==FluxBurgers :
    Tmax=3
while (tt[n] < Tmax):
    if n%5 == 0:
        plt.plot(x, uu[n,:],'b*-',x,uexact[n,:],'ro-')
        plt.xlabel('$x$',fontsize=11)
        plt.title(titre+', $t=$%s' %(tt[n]),fontsize=11)
        plt.legend(['Solution approcher','Solution exacte'], loc='best',fontsize=11)
        plt.pause(0.1)
    n+=1
plt.show()


#Tracage des courbe Caracteristique:-------------------------------------------

XX1,tt1=np.meshgrid(x,tt[:NT])
fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot(1,1,1, projection='3d')
contour=ax.plot_surface(XX1,tt1,uexact[:NT,:],rstride=4, cstride=4,alpha=1)
 
ax.set_xlabel('x',fontsize=24)
ax.set_ylabel('t',fontsize=24)
ax.view_init(60, 45)
fig.colorbar(contour)
fig.tight_layout()

fig.show()
 
fig1 = plt.figure(figsize=(18,6))
ax1 = fig1.add_subplot(1,1,1)
 
level=np.arange(uexact.min(),uexact.max(),0.005)
contour1=ax1.contour(XX1,tt1,uexact[:NT,:],level)
 
fig1.colorbar(contour1)
fig1.tight_layout()
ax1.set_xlabel('x',fontsize=24)
ax1.set_ylabel('t',fontsize=24)
plt.title('Courbes caracteristiques',fontsize=24)
fig1.show()

#------------------------------------------------------------------------------