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

def DI_significance_test(DI_ORIGINAL_MAT, DI_SURROGATES_MAT, delay_step,num_surs):
    """
    %DESCRIPTION: Function that estimates the Directional Information (DI)
    %through the significance of a single-trial statistic, denoted as 'T', which is defined as
    %T=max_{D} I_{D=1:max_delay}(X^N->Y^N),
    %where I_{D}(X^{t-D}->t^N) is the directed information measure (Massey 90) computed for a time
    %delay 'D' during a trial interval. 
    
    
    %INPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %1. DI_ORIGINAL_MAT: (num_trials x num_intervals x num_delays) matrix of DI
    %computations
    %2. DI_SURROGATES_MAT: (num_trials x num_intervals x num_delays x num_surs) matrix of DI
    %computations
    %3. delay_step: Difference in bins between consecutive tested delays. 
    %4. num_surs: number of surrogates to generate the null hypothesis
      
      
    %OUTPUTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %1. PVAL_MAT: (num_trials x num_intervals ) matrix of p-values for DI computations
    %2. DELAY_MAT: (num_trials x num_intervals) matrix of delays for DI
    %computations
    
    
    %REFERENCE:
    %A. Tauste Campo, Y. Vázquez, M. Àlvarez, A. Zainos, R. Rossi-Pool, G. Deco, R Romo. 
    %"Feed-forward information and zero-lag synchronization in the sensory thalamocortical circuit are modulated during stimulus perception", 
    %PNAS, 116(15), pp. 7513-22, 2019. (Supplementary Information)
    """ 
     
    # Construct the original value of T for each trial and interval
    T=DI_ORIGINAL_MAT.argmax(axis=-1)
    ind_delays=DI_ORIGINAL_MAT.max(axis=-1)
    
    # convert indeces into delays/lags (accounting for zero delay/lags).
    DELAY_MAT=delay_step*ind_delays
    
    # Construct the reference distribution H0 of T out of the surrogate estimates:
    H0=DI_SURROGATES_MAT.max(axis=-1)
    
    # Extract matrix dimensions
    num_trials=T.shape[0]
    num_intervals=T.shape[1]
    PVAL_MAT=np.zeros((num_trials, num_intervals))
    
    
    # Compute the p-value associated to the statistic T. 
    for trial in range(num_trials):
        for interval in range(num_intervals):
          PVAL_MAT[trial,interval]= ( 1 + sum(H0[trial,interval,:]>=T[trial,interval]) ) / (1+num_surs)

    
    return PVAL_MAT,DELAY_MAT, T, H0



 