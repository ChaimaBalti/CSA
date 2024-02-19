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
    creneau

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    N=np.size(x)
    y=np.zeros(N)
    mask1=((x>=-2)*(x<-1/2))+((x>1/2)*(x<=2))
    mask2=(x>=-1/2)*(x<1/2)
    y=1*mask2+(-0.5)*mask1 
    
    return y
    

def Rampe(x):
    '''
    Rampe

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    N=np.size(x)
    y=np.zeros(N)
    mask1=((x>=-2)*(x<-1/2))+((x>=1)*(x<=2))
    mask2=(x>=-1/2)*(x<1)
    y=x*mask2+(-0.5)*mask1 
    
    return y
    
def atild(U,V, flux):
    fU = flux(U, 0);
    fV = flux(V, 0);
    fprime= flux(U, 1);   
    if U==V :
        return fprime
    else :
        return (fU-fV)/(U-V)
    
def g(U,V, flux):
    a=atild(U,V, flux)
    mask=(a>=0)
    fU = flux(U, 0);
    fV = flux(V, 0);
    return mask*fU+(1-mask)*fV

def Decentre(U,V,W,dt_sur_h, flux):

    H=np.zeros([np.size(U),1]);
    
    fU = flux(U, 0);
    fV = flux(V, 0);
    fW = flux(W, 0);
    
    a=flux(V, 1);
    
    mask=(a>=0)

    H=mask*(V-dt_sur_h*(fV-fU))+(1-mask)*(V-dt_sur_h*(fW-fV))
    
    return H

def MurmanRoe(U,V,W,dt_sur_h, flux):

    H=np.zeros([np.size(U),1]);
    
    gVW = g(V,W, flux);
    gUV = g(U,V, flux);
    H=V-dt_sur_h*(gVW-gUV)
    return H

def LaxFriedrichs(U,V,W,dt_sur_h, flux):

    H=np.zeros([np.size(U),1]);

    fV = flux(V, 0);
    fW = flux(W, 0);

    H=0.5*((U+W)-dt_sur_h*(fW-fV))
    
    return H

def Godunov(U,V,W,dt_sur_h, flux):

    H=np.zeros([np.size(U),1]);
    
    return H
# Paramètres du probleme-------------------------------------------------------

#Le flux qui determine la lois de conservation a resoudre  f(u)
flux  = FluxTransport;  # % Flux pour l equation de transport.
#flux = FluxBurgers;     # Flux pour l equation de Burgers. 


#Condition intiale a utiliser
ConditionInitiale = Rampe;
#ConditionInitiale = Creneau;  
#ConditionInitiale = schoc;


# %Schema a utiliser (expression de H(u,v,w))
Hscheme = Decentre;         # (A coder)
# Hscheme = @MurmanRoe;        % (A coder)
# %Hscheme = @LaxFriedrichs;    % (A coder)
# %Hscheme = @Godunov;          % (A coder)


# %Discretisation spatiale
N = 200;

# %Discretisation temporelle
alpha = 0.8;

# %Variable valant 1 si l'on veut afficher la solution exacte
# %Attention, fonctionne pour :
# % -Tous les problemes de transports lineaire
# % -Burgers avec condition initiale rampe 
sol_exacte=1;


# %--------------------------------------------------------------------------
# %Le code qui suit ne doit pas etre modifier dans le cadre du TP
# %--------------------------------------------------------------------------

# %Domaine spatiale et temporel
xmin = -2;
xmax = 2;
Tmin = 0;
Tmax = 5;



# Discretisation spatial
h = (xmax-xmin)/N;
x=np.linspace(xmin, xmax-h, N);

# Initialisation du compteur d'itérations et du temps initial
n=0;
t=Tmin;


# Condition initiale
u=np.zeros([N,1])
u[:,0]= ConditionInitiale(x);


# Estimation du nombre maximale d'iterations en temps

dt = alpha * h / np.max( np.abs( flux(u[0,:],1) ) );
NT = int(np.floor( (Tmax - Tmin) / dt ));
tt=np.zeros([NT+2,1])
dtt=np.zeros([NT+2,1])
tt[0]=t
dtt[0]=dt
uu=np.zeros([NT+2,N])
uu[0,:]=u[:,0]

# Solution exacte
uexact=np.zeros([NT+2,N])
uexact[0,:]= np.copy(u[:,0])

# Estimation de la norme infinie de la solution
LinftyCondInit = np.max(np.abs(u))

# calcul du min et max pour regler les axes
umin = np.min(u[0,:])
umax = np.max(u[0,:])
du = np.abs(umax-umin)
umin = umin - 0.1*du
umax = umax + 0.1*du


#--------------------------------------------------------------------------
# boucle en temps
#--------------------------------------------------------------------------

U=np.zeros([N,1]) #j-1
V=np.zeros([N,1]) #j
W=np.zeros([N,1]) #j+1

while (tt[n] < Tmax):
    
    # Calcul du dt adaptatif
    #       alpha * h
    # dt = -----------------
    #       sup | a( u(n,:) ) |
    
    dt = alpha * h / np.max( np.abs( flux(uu[n,:],1) ) ); 
    
    dtt[n]=dt
    tt[n+1] = tt[n]+dtt[n];
    
    #U = np.array([uu[n,N-1], uu[n,0:N-1] ]);
    
    U[1:N,0]=uu[n,0:N-1];U[0,0]=uu[n,N-1]
    V[:,0] = uu[n,:]
    
    #W = np.array([uu[n,1:N], uu[n,0]]);
    
    W[0:N-1,0]=uu[n,1:N];W[N-1,0]=uu[n,0]
    
    # on applique le schema u_j^{n+1} = H(U,V,W)  
    u = Hscheme(U, V, W, dtt[n]/h, flux)
    
    uu[n+1,:]=u[:,0];  #Valeur approchée a t=n+1
    
    # calcul de la solution exacte.
    if (flux==FluxTransport):
        uexact[n+1,:] =  ConditionInitiale(xmin+np.mod(x-xmin-flux(0,1)*tt[n+1],(xmax-xmin)))
    
    
    # plt.plot(x,uu[n+1,:],'rs-',x,uexact[n+1,:],'b*-')
    # plt.pause(0.1)
    n=n+1;
    

plt.figure(1)
#matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif'})

for n in np.arange(0,NT+2):
    if n%10 == 0:
        plt.plot(x, uu[n,:],'b*-',x,uexact[n,:],'ro-')
        plt.xlabel('$x$',fontsize=11)
        plt.title('Schema explicite centre, $t=$%s' %(tt[n]),fontsize=11)
        plt.legend(['Solution approcher','Solution exacte'], loc='best',fontsize=11)
        plt.pause(0.1)
plt.show()