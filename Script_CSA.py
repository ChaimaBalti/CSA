# -*- coding: utf-8 -*-
"""
Created on Tue May  7 02:10:31 2024

@author: balti_j80n85d
"""

#-----------------------------Importation des Bibliothèques-------------------------

import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import root_scalar


#-----------------------Definition des parametres du probleme-------------------------

gamma = 1.4

#Cas n1
rho_G = 1
u_G = 1
rho_D = 4 
u_D = 4

#cas n2
rho_G = 1
u_G = 3
rho_D = 1 
u_D = 1

#-----------------------------Definition des fonctions-------------------------

def P(rho):
    return  rho**gamma  

def u_1_D(rho,GD='G'):
    '''
    Parameters
    ----------
    rho : TYPE
        DESCRIPTION.
    GD : TYPE, optional
        DESCRIPTION. The default is 'G' if U_G OR 'D' if U_D

    Returns
    -------
    TYPE
        DESCRIPTION.
    '''
    if GD=='D': #Droite
        return u_D + (2*np.sqrt(gamma))/(gamma-1)*( rho_D**((gamma-1)/2) - rho**((gamma-1)/2) )
    return u_G - ( (2*np.sqrt(gamma))/(gamma-1) )*( rho**((gamma-1)/2) - rho_G**((gamma-1)/2) )

def u_2_D(rho,GD='D'):
    '''
    Parameters
    ----------
    rho : TYPE
        DESCRIPTION.
    GD : TYPE, optional
        DESCRIPTION. The default is 'D' if U_D OR 'G' if U_G

    Returns
    -------
    TYPE
        DESCRIPTION.
    '''
    if GD=='G': #Gauche
        return u_G + (2*np.sqrt(gamma))/(gamma-1)*( rho**((gamma-1)/2) - rho_G**((gamma-1)/2) )
    return u_D + ((2*np.sqrt(gamma))/(gamma-1))*( rho**((gamma-1)/2) -  rho_D**((gamma-1)/2) )
                                     
def u_1_C(rho,GD='G'):
    '''
    Parameters
    ----------
    rho : TYPE
        DESCRIPTION.
    GD : TYPE, optional
        DESCRIPTION. The default is 'G' if U_G OR 'D' if U_D

    Returns
    -------
    TYPE
        DESCRIPTION.
    '''
    if GD=='D': #Droite
        return u_D - np.sqrt( (rho-rho_D)*(P(rho)-P(rho_D))/(rho*rho_D) )
    return u_G - np.sqrt( (rho-rho_G)*(P(rho)-P(rho_G))/(rho*rho_G) )
 
def u_2_C(rho,GD='D'):
    '''
    Parameters
    ----------
    rho : TYPE
        DESCRIPTION.
    GD : TYPE, optional
        DESCRIPTION. The default is 'D' if U_D OR 'G' if U_G

    Returns
    -------
    TYPE
        DESCRIPTION.
    '''
    if GD=='G': #Gauche
        return u_G + np.sqrt( (rho_G-rho)*(P(rho_G)-P(rho))/(rho_G*rho) )
    return u_D + np.sqrt( (rho_D-rho)*(P(rho_D)-P(rho))/(rho_D*rho) )

def C1_C2(rho, case=1):
    '''
    
    Parameters
    ----------
    rho : TYPE
        DESCRIPTION.
    case : TYPE, optional
        DESCRIPTION. The default is 1. 

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if case == 2:
        C1=u_1_C(rho,'D')*(rho > rho_D) + u_1_D(rho,'D')*(rho <= rho_D)
        C2=u_2_C(rho,'G')*(rho > rho_G) + u_2_D(rho,'G')*(rho <= rho_G)
    else:
        C1=u_1_C(rho)*(rho > rho_G) + u_1_D(rho)*(rho <= rho_G)
        C2=u_2_C(rho)*(rho > rho_D) + u_2_D(rho)*(rho <= rho_D)
    return C1-C2


rho_min=0
rho_max=5

rho_supG=np.linspace(rho_G, rho_max, 100)
rho_infG=np.linspace(rho_min, rho_G, 100)
rho_supD=np.linspace(rho_D, rho_max, 100)
rho_infD=np.linspace(rho_min, rho_D, 100)

plt.plot(rho_supG,u_1_C(rho_supG),'r' )
plt.plot(rho_infG,u_1_D(rho_infG),'b')

plt.plot(rho_supD,u_2_C(rho_supD),'m')
plt.plot(rho_infD,u_2_D(rho_infD),'orange')

plt.grid()

rho_et1=root_scalar(lambda rho : C1_C2(rho,1), bracket=[0.1, 6])['root']
U_et1=u_1_D(rho_et1)
plt.plot(rho_et1,U_et1,'og') 

plt.plot(rho_G,u_G,'xm')
plt.plot(rho_D,u_D,'xb')

eps=10**(-3.5)
rho_et2=root_scalar(lambda rho : C1_C2(rho,2), bracket=[0.1, 6])['root']
U_et2=u_2_C(rho_et2,'G')
#plt.plot(rho_et2,U_et2,'ob')
plt.legend(['1C','1D','2C','2D','U*','(rho_G,u_G)','(rho_D,u_D)'])
plt.show()

#---------------------------Calcul de la solution exacte-------------------------
t=0.2
x=np.linspace(-2,2,100)

def lamda(u,rho,n=1):
    '''
    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    n : TYPE, optional
        DESCRIPTION. The default is 1 pour lambda1 et 2 pour lambda2

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return ( u-np.sqrt(gamma)*( rho**((gamma-1)/2) ) )*(n==1) + ( u+np.sqrt(gamma)*( rho**((gamma-1)/2) ) )*(n==2)
  
s1=u_G-((1/rho_G)*((P(rho_G)-P(rho_et1))/(U_et1-u_G)))
s2=u_D-((1/rho_D)*((P(rho_et1)-P(rho_D))/(u_D-U_et1)))
 
def U_1_det(x,t):
    u=  (2/(gamma+1)) * (x/t) + ((gamma-1)/(gamma+1))*u_G   +(2/(gamma+1))*np.sqrt(gamma*rho_G**(gamma-1))
    rho= ((u-(x/t))/np.sqrt(gamma))**(2/(gamma-1))
    return rho,u

def U_2_det(x,t):
    u=  (2/(gamma+1)) * (x/t) + ((gamma-1)/(gamma+1))*u_D   -(2/(gamma+1))*np.sqrt(gamma*rho_D**(gamma-1))
    rho= (((x/t)-u)/np.sqrt(gamma))**(2/(gamma-1))
    return rho,u
 
def u(x,t,L1='choc',L2='choc'):
    '''

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    L1 : TYPE, optional
        DESCRIPTION. The default is 'choc'.
    L2 : TYPE, optional
        DESCRIPTION. The default is 'choc'.

    Returns
    -------
    None.

    '''
    if L1=='choc' and L2=='choc':
        u=u_G*( x/t < s1 ) + U_et1*((x/t > s1)*(x/t < s2)) + u_D*(x/t > s2)
        rho=rho_G*( x/t < s1 ) + rho_et1*((x/t > s1)*(x/t < s2)) + rho_D*(x/t > s2)
            
    elif L1=='detente' and L2=='choc':
        u=u_G*( x/t < lamda(u_G, rho_G)  ) + U_1_det(x,t)[1]*((x/t >= lamda(u_G, rho_G))*( x/t <= lamda(U_et1, rho_et1))) + U_et1*(( x/t >= lamda(U_et1, rho_et1))*(x/t < s2)) + u_D*(x/t > s2)
        rho=rho_G*( x/t < lamda(u_G, rho_G)  ) + U_1_det(x,t)[0]*((x/t >= lamda(u_G, rho_G))*( x/t <= lamda(U_et1, rho_et1))) + rho_et1*(( x/t >= lamda(U_et1, rho_et1))*(x/t < s2)) + rho_D*(x/t > s2)
        
    elif L1=='choc' and L2=='detente' :
        u=u_G*( x/t < s1 ) + U_et1*(( x/t <= lamda(U_et1, rho_et1,2))*(x/t > s1)) + U_2_det(x,t)[1]*((x/t >= lamda(U_et1, rho_et1,2))*( x/t <= lamda(u_D, rho_D,2)))  + u_D*( x/t >= lamda(u_D, rho_D,2))
        rho=rho_G*( x/t < s1 ) + rho_et1*(( x/t <= lamda(U_et1, rho_et1,2))*(x/t > s1)) + U_2_det(x,t)[0]*((x/t >= lamda(U_et1, rho_et1,2))*( x/t <= lamda(u_D, rho_D,2)))  + rho_D*( x/t >= lamda(u_D, rho_D,2))
    
    elif L1=='detente' and L2=='detente' :
        u=u_G*( x/t < lamda(u_G, rho_G)  ) + U_1_det(x,t)[1]*((x/t >= lamda(u_G, rho_G))*( x/t <= lamda(U_et1, rho_et1))) + U_et1*(( x/t <= lamda(U_et1, rho_et1,2))*(x/t >= lamda(U_et1, rho_et1,1))) + U_2_det(x,t)[1]*((x/t >= lamda(U_et1, rho_et1,2))*( x/t <= lamda(u_D, rho_D,2)))  + u_D*( x/t >= lamda(u_D, rho_D,2))
        rho=rho_G*( x/t < lamda(u_G, rho_G)  ) + U_1_det(x,t)[0]*((x/t >= lamda(u_G, rho_G))*( x/t <= lamda(U_et1, rho_et1))) + rho_et1*(( x/t <= lamda(U_et1, rho_et1,2))*(x/t >= lamda(U_et1, rho_et1,1))) + U_2_det(x,t)[0]*((x/t >= lamda(U_et1, rho_et1,2))*( x/t <= lamda(u_D, rho_D,2)))  + rho_D*( x/t >= lamda(u_D, rho_D,2))
    
    U = np.zeros([2,np.size(x)])
    U[0,:] = rho
    U[1,:] = u
    return U


#  Validation de la fonction u(x,t,'_','_')
    
xmin=-2
xmax=2
Nx=1000
L=xmax-xmin
dx=L/Nx
X=np.linspace(xmin,xmax+dx,Nx)

Tmax=0.3
Nt=100
t=np.linspace(0,Tmax,Nt)

U_exact=u(X,0.2,'detente','detente')

plt.plot(X,U_exact[0,:],'r*-')
plt.xlabel('x')
plt.ylabel('rho')
plt.show()

plt.plot(X,U_exact[1,:],'r*-')
plt.xlabel('x')
plt.ylabel('u')
plt.show()

SolutionAuCoursDuTemps=0
if SolutionAuCoursDuTemps :
    champs_1='choc'
    champs_2='choc'
    rho_t=np.zeros([Nt,Nx])
    u_t=np.zeros([Nt,Nx])
    for i in range(0,Nt):
        ti=t[i]
        U_exact=u(X,ti,champs_1,champs_2)
        rho_t[i,:]=U_exact[0,:]
        u_t[i,:]=U_exact[1,:]
        plt.subplot(1,2,1)
        plt.plot(X,rho_t[i,:],'g*-')
        plt.xlabel('x')
        plt.ylabel('rho')
        plt.subplot(1,2,2)
        plt.plot(X,u_t[i,:],'m*-')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.suptitle("Solution 1-"+champs_1+" 2-"+champs_2)
        plt.pause(0.1)
        plt.show()

#---------------------------Calcul de la solution approchée-------------------------

#  Calcul du pas de temps à chaque itération :
alpha=0.45
def Deltatn(W):
    rho=W[0,:]
    u=W[1,:]/rho
    m1=np.max(np.abs(lamda(u,rho,1)))
    m2=np.max(np.abs(lamda(u,rho,2)))
    return alpha*(dx/max(m1,m2))

# Validation de la fonction Deltatn
U_exact=u(X,0.2)
W=np.zeros([2,Nx])
W[0,:]=U_exact[0,:]
W[1,:]=U_exact[0,:]*U_exact[1,:]
Deltat=Deltatn(W)


#******************Definition du flux et des Schemas numerique********************
# Flux physique
def FluxPhy(W):
    fW=np.zeros([2,Nx])
    fW[0,:]=W[1,:]
    fW[1,:]=(W[1,:]**2/W[0,:])+(W[0,:]**gamma)
    return fW

# Flux numérique de Lax-Friederichs
def FluxG_LF(a,b,dtn):
    '''Cette fonction permet de calculer le flux numérique au point (j,j+1)
    a : matrice de taille 2*Nx, elle contient la valeur de W_j^n
    b : matrice de taille 2*Nx, elle contient la valeur de W_(j+1)^n
    dtn : le pas à l'instant tn
    La sortie G est une matrice de taille 2*Nx qui représente le flux numérique de L-F'''
    fa=FluxPhy(a)
    fb=FluxPhy(b)
    return ((fa+fb)/2.)-((dx/dtn)*(b-a)/2.)


# Flux numérique de Rusanov
def FluxG_Rusanov(a,b,dtn):
    '''Cette fonction permet de calculer le flux numérique au point (j,j+1)
    a : matrice de taille 2*Nx, elle contient la valeur de W_j^n
    b : matrice de taille 2*Nx, elle contient la valeur de W_(j+1)^n
    dtn : le pas à l'instant tn
    La sortie G est une matrice de taille 2*Nx qui représente le flux numérique de Rusanov'''
    fa=FluxPhy(a)
    fb=FluxPhy(b)
    c = np.max(np.array([np.abs(lamda(a[0,:],a[1,:],1)), np.abs(lamda(a[0,:],a[1,:],2)), np.abs(lamda(b[0,:],b[1,:],1)), np.abs(lamda(b[0,:],b[1,:],2))]),axis=0)
    return ((fa+fb)/2.)-(c*(b-a)/2.)

# Flux numérique de HLL
def FluxG_HLL(a,b,dtn):
    '''Cette fonction permet de calculer le flux numérique au point (j,j+1)
    a : matrice de taille 2*Nx, elle contient la valeur de W_j^n
    b : matrice de taille 2*Nx, elle contient la valeur de W_(j+1)^n
    dtn : le pas à l'instant tn
    La sortie G est une matrice de taille 2*Nx qui représente le flux numérique de HLL'''
    fa=FluxPhy(a)
    fb=FluxPhy(b)
    c1 = np.min(np.array([lamda(a[0,:],a[1,:],1), lamda(a[0,:],a[1,:],2), lamda(b[0,:],b[1,:],1), lamda(b[0,:],b[1,:],2)]),axis=0)
    c2 = np.max(np.array([lamda(a[0,:],a[1,:],1), lamda(a[0,:],a[1,:],2), lamda(b[0,:],b[1,:],1), lamda(b[0,:],b[1,:],2)]),axis=0)
    return fa*(c1>=0)+( ((c2*fa-c1*fb)/(c2-c1)) + ((b-a)*c1*c2)/(c2-c1) )*((c1<0)*(c2>0))+fb*(c2<=0)

#********************************Boucle Temps**********************************

# Initialisation de W
U0=u(X,0)
W=np.zeros([2,Nx])
W[0,:]=U0[0,:]
W[1,:]=U0[0,:]*U0[1,:]

t=0
Ntmax=1000

rho_app=np.zeros([Ntmax,Nx])
u_app=np.zeros([Ntmax,Nx])

rho_exact=np.zeros([Ntmax,Nx])
u_exact=np.zeros([Ntmax,Nx])

#rho et u approchés à t=0
rho_app[0,:]=U0[0,:]
u_app[0,:]=U0[1,:]
rho_exact[0,:]=U0[0,:]
u_exact[0,:]=U0[1,:]

n=0
tn=[0]

Sol_app_temps=1
if Sol_app_temps :
    #flux_numerique=FluxG_LF ;Schema= "Schema de Lax-Friederichs"
    #flux_numerique=FluxG_Rusanov ;Schema= "Schema de Rusanov"
    flux_numerique=FluxG_HLL ;Schema= "Schema de HLL"

    champs_1='choc'
    champs_2='choc'       #'choc' 'detente'
    dtn=Deltatn(W)
    while (t<Tmax):
        #Calcul de deltatn
        if not(np.isnan(Deltatn(W)) ) :
            dtn=Deltatn(W) 
        #print(dtn)
        t+=dtn
        tn.append(t)
        d=dtn/dx
        Wj=W
        vect_D=np.array([[rho_D],[rho_D*u_D]])
        Wjp1=np.hstack((W[:,1:Nx],vect_D))
        vect_G=np.array([[rho_G],[rho_G*u_G]])
        Wjm1=np.hstack((vect_G,W[:,0:Nx-1]))
        W=W-d*(flux_numerique(Wj,Wjp1,dtn)-flux_numerique(Wjm1,Wj,dtn))
        n+=1
        
        rho_app[n,:]=W[0,:]
        u_app[n,:]=W[1,:]/W[0,:]
        U_exact=u(X,t,champs_1,champs_2)
        rho_exact[n,:]=U_exact[0,:]
        u_exact[n,:]=U_exact[1,:]
        
        if n%300 == 0 :
            plt.subplot(1,2,1)
            plt.plot(X,rho_app[n,:],'g*-')
            plt.plot(X,rho_exact[n,:],'y+-')
            plt.xlabel('x')
            plt.ylabel('rho')
            plt.legend(['rho approchée','rho exact'])
            
            plt.subplot(1,2,2)
            plt.plot(X,u_app[n,:],'m*-')
            plt.plot(X,u_exact[n,:],'b+-')
            plt.xlabel('x')
            plt.ylabel('u')
            plt.legend(['u approché','u exact'])
            
            plt.pause(0.001)
            plt.show()
            plt.suptitle("Solution 1-"+champs_1+" 2-"+champs_2+": "+Schema)

