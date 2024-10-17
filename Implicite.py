#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on FEV 13 2024

@author: Chaima Balti 2AMIndS
"""
#******************************************************************************
# Résolution numérique de l'équation de transport
# u,t + c u,x = 0
# avec des schémas implécite
#******************************************************************************

# Load packages----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt 
import scipy as sp
from scipy.sparse import csc_matrix

# Paramètres du probleme-------------------------------------------------------


# Parametres de discretisation
h  = 0.01
dt = 0.01

# Condition initiale
#   0 => gaussienne
#   1 => chapeau
cond_ini =0

# Vitesse
#   0 => constante  (cas i)
#   1 => variable   (cas ii)
vitesse = 1


# Creation des maillages-------------------------------------------------------

x = np.arange(0,1,h);   # Le point d'indice Nx-1 correspond a x=1-h, identique a x=h d'apres les CL
Nx =len(x);             # Le point d'indice Nx correspond a x=1, identique a x=0 d'apres les CL


t = np.arange(0,1+dt,dt);
Nt = len(t);

# Initialisation---------------------------------------------------------------

U = np.zeros([Nt, Nx]);


def gaussienne(x):
    '''% (avec 0 <= x <= 1)
    Fonction permettant de construire une solution initiale  en gaussienne:
        '''
    sigma2 = 0.01;
    a=0.1
    u=a*np.exp((-(x-1/2)**2)/sigma2)
    return u

def chapeau(x):
    '''% (avec 0 <= x <= 1)
    Fonction permettant de construire une solution initiale  en "chapeau":
        '''
    N=np.size(x)
    y=np.zeros(N)
    mask1=(x<1/4)
    mask2=(x>=1/4)*(x<=1/2)
    y=x*mask1+(0.5-x)*mask2  
    return y
        

if  cond_ini==0:
    U[0,:]=gaussienne(x);   
else: 
    U[0,:]=chapeau(x);


# Boucle en temps--------------------------------------------------------------

print(U)
for n in np.arange(0,Nx):
    if vitesse == 0:
        c=0.8                # cas (i)
    else:
        c=np.sin(10*t[n+1])    # cas (ii)
    alpha = c*(dt/h)
    
    A = csc_matrix(np.diag((1+alpha)*np.ones(Nx-1),k=1)+np.diag((1-alpha)*np.ones(Nx)))
    A[Nx-1,0]=1+alpha
    B = csc_matrix(np.diag((1-alpha)*np.ones(Nx-1),k=1)+np.diag((1+alpha)*np.ones(Nx)))
    B[Nx-1,0]=1-alpha
    
    U[n+1,:]=sp.sparse.linalg.spsolve(A,B.dot(U[n,:]))


#------------------------------------------------------------------------------
# Le code qui suit ne doit pas etre modifier dans le cadre du TP   

# Solution exacte
# Uex(t,X(t)) = U0(X(0)) = U0(x-d)
# avec
# d = X(t) - X(0) = \int_0^t c(t) dt      



Uex = np.zeros([Nt, Nx]);
err = np.zeros([Nt, Nx]);
err2 = np.zeros([Nt, 1]);
for n in np.arange(0,Nt):
    for j in np.arange(0,Nx):
        if vitesse == 0:
            d=c*t[n]
        else:
            d=0.1*(1.-np.cos(10*t[n]))
        
        if cond_ini == 0:
            Uex[n,j]=gaussienne(np.mod(x[j]-d,1));
        else:
            Uex[n,j]=chapeau(np.mod(x[j]-d,1));
            
            
        err[n,j] = U[n,j] - Uex[n,j];
    err2[n] = np.linalg.norm(err[n,:], 2);
        
err_inf_2 = np.max(err2)            


plt.figure(1)
#matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif'})

for n in np.arange(0,Nt):
    if n%3 == 0:
        plt.plot(x, U[n,:],'b*-',x,Uex[n,:],'ro-')
        plt.xlabel('$x$',fontsize=11)
        plt.title('Schema explicite centre, $t=$%s' %(t[n]),fontsize=11)
        plt.legend(['Solution approcher','Solution exacte'], loc='best',fontsize=11)
        plt.pause(0.1)
plt.show()

XX,tt=np.meshgrid(x,t)
fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot(1,1,1, projection='3d')
contour=ax.plot_surface(XX,tt,U,rstride=4, cstride=4,alpha=1)
#contour=ax.plot_trisurf(XX,tt,U)
#contour=ax.contourf(XX,tt,U)

#fig.tight_layout()
ax.view_init(45, 45)
fig.colorbar(contour)
fig.tight_layout()

fig.show()
#plt.show()


fig1 = plt.figure(figsize=(18,6))
ax1 = fig1.add_subplot(1,1,1)

#contour=ax.plot_surface(XX,tt,U)
level=np.arange(U.min(),U.max(),0.005)
contour1=ax1.contour(XX,tt,U,level)
#contour1=ax1.contour(XX,tt,U,cmap=matplotlib.cm.RdBu,vmin=U.min(),vmax=U.max())

#ax1.title(['Courbes caracteristiques'])

#ax1.xlabel('x')
#ax1.ylabel('t')
#fig.tight_layout()
#ax1.view_init(45, 45)
fig1.colorbar(contour1)
fig1.tight_layout()
ax1.set_xlabel('x',fontsize=24)
ax1.set_ylabel('t',fontsize=24)
plt.title('Courbes caracteristiques',fontsize=24)
fig1.show()


