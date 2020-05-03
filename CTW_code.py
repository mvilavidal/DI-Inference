"""
%Copyright (C) 2020, Manel Vila-Vidal, for the translation from Matlab to Python
Contact details: m@vila-vidal.com
%Copyright (C) 2019, Adria Tauste Campo, for the adapted Matlab code (https://github.com/AdTau/DI-Inference)
%Contact details: adria.tauste@gmail.com
%Copyright (C) 2019, Jiantao Jiao, for the original Matlab code (https://github.com/EEthinker/Universal_directed_information)

%This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License v2.0 as published by the Free Software Foundation.
%This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License v2.0 for more details.
%You should have received a copy of the GNU General Public License v2.0 along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import numpy as np


def  compute_DI_M(X,Y,Nx,D,Dxy,alg,start_ratio,prob,flag):
    """
    %DESCRIPTION:
    % Function `compute_DI_MI' calculates the directed information I(X^n-->
    % Y^n), mutual information I(X^n; Y^n) and reverse directed information I(Y^{n-1}-->X^n)
    % for any positive integer n smaller than the length of X and Y. 
    
    % X and Y: two input sequences;
    % Nx:  the size of alphabet of X, assuming X and Y have the same size of
    % alphabets;
    % D:  the maximum depth of the context tree used in basic CTW algorithm,
    % for references please see F. Willems, Y. Shtarkov and T. Tjalkens, 'The
    % Context-Tree Weighting Method: Basic Properties', IEEE Transactions on
    % Information Theory, 653-664, May 1995.
    % alg:  indicates one of the four possible estimators proposed in J.
    % Jiao. H. Permuter, L. Zhao, Y.-H. Kim and T. Weissman, 'Universal
    % Estimation of Directed Information', http://arxiv.org/abs/1201.2334.
    % Users can indicate strings 'E1','E2','E3' and 'E4' for corresponding
    % estimators.
    """

    # map the data pair (X,Y) into a single variable taking value with alphabet size |X||Y|
    
    XY=X+Nx*Y
    
    if flag == 0:
        # Calculate the CTW probability assignment
        pxy = ctwalgorithm_M(XY,Nx**2,Dxy)
        px = ctwalgorithm_M(X,Nx,D)
        py = ctwalgorithm_M(Y,Nx,D)
    else:
        pxy = prob['pxy']
        px = prob['px']
        py = prob['py']

    # px_xy is a Nx times n_data matrix, calculating p(x_i|x^{i-1},y^{i-1})
    #pxy=pxy[:,pxy.shape[1]-px.shape[1]+1:] # ---> APAÑO!!!!
    px_xy=np.zeros(px.shape)
    for i_x in range(Nx):
        px_xy[i_x,:]=pxy[i_x,:]
        for j in range(1,Nx):
            px_xy[i_x,:]+=pxy[i_x+j*Nx,:]

    # calculate P(y|x,X^{i-1},Y^{i-1})
    temp= np.tile(px_xy,(Nx,1))
    py_x_xy=pxy/temp

    if alg=='E1': # use the trick that px(2,1)=px(2), px(2,2)=px(2+Nx)
        raise  Exception('Option not implemented')
        # temp_MI=-log2(px(X(D+1:end)+[1:Nx:end-Nx+1])) - log2(py(Y(D+1:end)+[1:Nx:end-Nx+1]))+  log2(pxy(XY(D+1:end)+[1:Nx^2:end-Nx^2+1]))  ;
        # temp_DI= - log2(py(Y(D+1:end)+[1:Nx:end-Nx+1]))+  log2(pxy(XY(D+1:end)+[1:Nx^2:end-Nx^2+1])) -log2(px_xy(X(D+1:end)+[1:Nx:end-Nx+1]))  ;
        # temp_rev_DI=-log2(px(X(D+1:end)+[1:Nx:end-Nx+1]))+log2(px_xy(X(D+1:end)+[1:Nx:end-Nx+1])) ;
        
    
    elif alg=='E2':
        raise  Exception('Option not implemented')
        #temp_MI=ctwentropy(px)+ctwentropy(py)-ctwentropy(pxy);
        #temp_DI=ctwentropy(py)-ctwentropy(pxy)+ctwentropy(px_xy);
        #temp_rev_DI=ctwentropy(px)-ctwentropy(px_xy);
    
    elif alg=='E3':
        #temp_MI=np.zeros(px.shape[1])
        #temp_DI= temp_MI
        #temp_rev_DI=temp_MI
        temp_DI=np.zeros(px.shape[1])
        for iy in range(Nx):
            ind = X[D:] + iy*Nx+range(0,py_x_xy.shape[0]-Nx**2+1,Nx**2)
            temp_DI = temp_DI + py_x_xy[ind,:] * np.log2( py_x_xy[ind]/py[iy,:] )

    
    elif alg=='E4':
        raise  Exception('Option not implemented')
        # temp_MI=zeros(1,size(px,2));
        # temp_DI= temp_MI;
        # temp_rev_DI=temp_MI;
        # for iy=1:Nx
        #    for ix=1:Nx
        #        temp_MI=temp_MI+pxy(ix+(iy-1)*Nx,:).*log2(pxy(ix+(iy-1)*Nx,:)./(py(iy,:).*px(ix,:)));
        #        temp_DI=temp_DI+pxy(ix+(iy-1)*Nx,:).*log2(pxy(ix+(iy-1)*Nx,:)./(py(iy,:).*px_xy(ix,:)));
        #        temp_rev_DI=temp_rev_DI+pxy(ix+(iy-1)*Nx,:).*log2(px_xy(ix,:)./px(ix,:));
    
    DI = temp_DI    
    return DI



def ctwalgorithm_M(x,Nx,D):
    """
    % DESCRIPTION: Function CTWAlgorithm outputs the universal sequential probability
    % assignments given by CTW method.
    """
    
    if x.ndim != 1: 
        raise Exception('The input must be a vector!')
    
    n=x.shape[0]
    #countTree = zeros(Nx, (Nx^(D+1) - 1) / (Nx-1)) ;
    #betaTree = ones(1,(Nx^(D+1) - 1 )/ (Nx-1))  ;
    Px_record = np.zeros((Nx,n-D))
    indexweight = Nx**np.arange(D)
    offset = (Nx**D - 1) / (Nx-1) + 1
    ####
    index_M=[1]
    contT_M=[np.zeros(Nx)]
    beta_M=[1.]
    ####
    for i in range(D,n):
        context = x[i-D:i]
        leafindex = np.sum(context*indexweight)+offset
        xt = x[i]
        
        ### si no existeix cami el creem
        if leafindex not in index_M:
            index_M+=[leafindex]
            beta_M+=[1.]
            contT_M+=[np.zeros(Nx)]
            j_index=len(index_M)-1
        else:
            j_index=index_M.index(leafindex)

        eta = (contT_M[j_index][:Nx-1]+0.5) / (contT_M[j_index][-1]+0.5)
        # update the leaf
        contT_M[j_index][xt]+=1
        node=int((leafindex+Nx-2)/Nx)
       
        ### si no existeix cami el creem
        if node not in index_M:
            index_M+=[node]
            beta_M+=[1.]
            contT_M+=[np.zeros(Nx)]
            j_index=len(index_M)-1
        else:
            j_index=index_M.index(node)  

        while node!=0:
            eta,beta_M,contT_M = ctwupdate_M(eta, node, xt,1/2,beta_M,contT_M, j_index)
            node =int((node+Nx-2)/Nx)
            # si no existeix cami el creem
            if node not in index_M:
                index_M+=[node]
                beta_M+=[1.]
                contT_M+=[np.zeros(Nx)]
                j_index=len(index_M)-1
            else:
                j_index=index_M.index(node)


        eta_sum = sum(eta)+1
        Px_record[:,i-D] = np.concatenate((eta,[1]))/eta_sum
    
    return Px_record





def ctwupdate_M(eta, index, xt,alpha,beta_M,contT_M, j_index):
    """
    % countTree:  countTree(a+1,:) is the tree for the count of symbol a a=0,...,M
    % betaTree:   betaTree(i(s) ) =  Pe^s / \prod_{b=0}^{M} Pw^{bs}(x^{t})
    % eta = [ p(X_t = 0|.) / p(X_t = M|.), ..., p(X_t = M-1|.) / p(X_t = M|.)
    
    % calculate eta and update beta a, b
    % xt is the current data
    """
    
    #size of the alphbet
    Nx = eta.shape[0]+1
    
    pw = np.concatenate((eta,[1]))
    pw = pw/pw.sum()  # pw(1) pw(2) .. pw(M+1)
    
    pe = (contT_M[j_index]+0.5) / (sum(contT_M[j_index])+Nx/2)
    temp= beta_M[j_index]
    
    if (temp < 1000):
        eta = (alpha*temp * pe[0:-1] + (1-alpha)*pw[0:-1]) / ( alpha * temp*pe[-1] + (1-alpha)*pw[-1]) 
    else:
        eta = (alpha*pe[0:-1] + (1-alpha)*pw[0:-1]/temp ) / ( alpha*pe[-1] + (1-alpha)*pw[-1]/temp) 
    
    beta_M[j_index]=beta_M[j_index] * pe[xt]/pw[xt]
    contT_M[j_index][xt]+=1
    
    return eta,beta_M,contT_M
