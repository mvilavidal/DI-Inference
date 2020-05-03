"""
%Copyright (C) 2020, Manel Vila-Vidal, for the translation from Matlab to Python
Contact details: m@vila-vidal.com
%Copyright (C) 2019, Adria Tauste Campo, for the original Matlab code (https://github.com/AdTau/DI-Inference)
%Contact details: adria.tauste@gmail.com

%This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License v2.0 as published by the Free Software Foundation.
%This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License v2.0 for more details.
%You should have received a copy of the GNU General Public License v2.0 along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""


import numpy as np
import CTW_code as ctw


def DI_computation_per_pair(NODE_1_MAT,NODE_2_MAT,N,M,interval_width,interval_step,num_delays, delay_step, min_shift, max_shift, num_surs):
    """
    %DESCRIPTION: Function that estimates the directed information measure (Massey, 90) via the CTW algorithm (Jiao et al., 2013) at different
    %delays  betweeen simultaneously discrete signals. The code is structured to repeat this computation over all possible slicing time windows 
    %('intervals) of a given length within a fixed trial, and over all available trials. The function also performs the same computations for surrogate samples obtained for each parameter (delay, interval) by
    %shifting circulary the target sequence ('Y').
    
    %INPUTS
    %1.NODE_1_MAT-> (num_trials x trial length) signal matrix of node 1
    %2.NODE_2_MAT-> (num_trials x trial length) signal matrix of node 2
    %3.N->Alphabet size of the time series
    %4.M->Markovian memory of the time time series
    %5.interval_width->size of interval in time steps.
    %6.interval_steps->size of interval in time steps.
    %7.num_delays->number of delays to be tested starting from delay=0.
    %8.delay_step->number of time steps between tested delays
    %9. min_shift->minimim circular shift of the target sequence to obtain a surrogate estimation
    %10. max_shift->minimim circular shift of the target sequence to obtain a surrogate estimation
    %11.num_surs->number of equispaced surrogate estimations (one for each cyclic shift) 
    
    %OUTPUTS:
    %1.DI_ORIGINAL_MAT-> (num_trials x num_intervals x num_delays) matrix of DI
    %computations
    %2.DI_SURROGATES_MAT-> (num_trials x num_intervals x num_delays x num_surs) matrix of DI
    %computations
    
    %NOTE: The function 'compute_DI_M' and the sub-functions called therein ('CTW_code' subfolder) are based on the MATLAB implementation of the 
    %universal directed information estimators in Jiantao Jiao, Haim H. Permuter, Lei Zhao, Young-Han Kim, 
    %and Tsachy Weissman. "Universal estimation of directed information." IEEE Transactions on Information Theory 59, no. 10 (2013): 6220-6242. 
    %and slightlyu modifications of the Github code that can be found here:
    %https://github.com/EEthinker/Universal_directed_information
    
    
    %REFERENCE:
    %A. Tauste Campo, Y. Vázquez, M. Àlvarez, A. Zainos, R. Rossi-Pool, G. Deco, R Romo. 
    %"Feed-forward information and zero-lag synchronization in the sensory thalamocortical circuit are modulated during stimulus perception", 
    %PNAS, 116(15), pp. 7513-22, 2019. (Suplmentary Information)
    """

    # Set surrogate circular shifts%%%%%%%%%%%%%%%%%%%%%%%%%
    # the whole interval lasts 250 time steps so here I chose 
    # to start the shifts at 0.2 of the whole time length
    if num_surs <= max_shift-min_shift:
        sur_shifts=np.linspace(min_shift,max_shift,num_surs).astype(int)
    else:
        msg = 'Number of surrogates exceeds maximum possible'
        raise Exception(msg)
        
    
    ############### CTW ALGORITHM PARAMETERS ###############
    # Alphabet
    Nx=N
    # Markovian memory
    autocovariance_memory=M
    crosscovariance_memory=M
    # specific algorithm paramters
    start_ratio=0.5
    prob=0
    flag=0
    alg='E3'
    ###
    
    ############### Extract matrix dimensions and set intervals to consider ###############
    num_trials=NODE_1_MAT.shape[0]
    trial_length=NODE_1_MAT.shape[1]
    intervals=np.arange(interval_width,trial_length+1,interval_step) 
    num_intervals=intervals.shape[0]
    
    # Initialize matrices
    DI_ORIGINAL_MAT=np.zeros((num_trials,  num_intervals, num_delays))
    DI_SURROGATES_MAT=np.zeros((num_trials, num_intervals, num_delays, num_surs))
    
    
    # Trial Loop %%%%%%%%%%%%%%%%%%%%%%%%%
    for trial in range(num_trials):
    
        # Extract sequences per trial
        X=NODE_1_MAT[trial,:]
        Y=NODE_2_MAT[trial,:]
    
        # Interval Loop %%%%%%%%%%%%%%%%%%%%%%%%%
        for k in range(num_intervals):
              
            # Delay loop %%%%%%%%%%%%%%%%%%%
            cont_delay=0  
            
            for delay in range(0,num_delays*delay_step,delay_step):
                
                # Truncate binary sequences to account for delays
                ini_x=intervals[k]-interval_width
                fi_x=intervals[k]-delay
                ini_y=intervals[k]-interval_width+delay
                fi_y=intervals[k]
                Xseq=X[ini_x:fi_x]
                Yseq=Y[ini_y:fi_y]
                #  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
              
                # DIRECTED INFORMATION computation and storage
                DI=ctw.compute_DI_M(Xseq,Yseq,Nx,autocovariance_memory,crosscovariance_memory,alg,start_ratio,prob,flag)
                var=DI[-int(interval_width/2)-1:].mean() # average over second half of the interval
                DI_ORIGINAL_MAT[trial,k,cont_delay]=var
        
                # Surrogate computation loop%%%%%%%%%%%%%%%%%%%%%%
                for sur in range(num_surs):
        
                     #circular shifted surrogate sequence
                     Yseq_sur=np.roll(Yseq,sur_shifts[sur])
                     
                     # surrogate DI computation and storage                       
                     DI=ctw.compute_DI_M(Xseq,Yseq_sur,Nx,autocovariance_memory,crosscovariance_memory,alg,start_ratio,prob,flag); 
                     var=DI[-int(interval_width/2)-1:].mean() #average over second half of the interval
                     DI_SURROGATES_MAT[trial,k,cont_delay,sur]=var                                      
                
                cont_delay+=1
            
    return DI_ORIGINAL_MAT, DI_SURROGATES_MAT
          

