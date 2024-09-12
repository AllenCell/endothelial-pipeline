import numpy as np
from numpy.fft import fft, ifft
from scipy.signal import correlate
from scipy.stats import kurtosis

def next_pow_two(n):
    '''Helper function for FFT (finds nearest power of 2 greater than n)'''
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = fft(x - np.mean(x), n=2*n)
    acf = ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n
    
    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def correlate_1d(x, y, norm=True):
    '''1D correlation function for two time series (normalized by default)'''
    cf = correlate(x-np.mean(x), y-np.mean(y), mode='same')
    if norm:
        cf /= cf[0]
    return cf

def autocorr(x, norm=True):
    '''Autocorrelation function for any dimension'''
    if x.ndim == 1:
        return autocorr_1d(x,norm=norm)
    else:
        acf_mat = np.zeros((x.shape[0],x.shape[1],x.shape[1]))
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                if i==j:
                    acf_mat[:,i,j] = autocorr_1d(x[:,i], norm=norm)
                else:
                    acf_mat[:,i,j] = correlate_1d(x[:,i], x[:,j], norm=norm)
        return acf_mat

# Mean kurtosis of dx/dt over spatial grid with given bin edges at given lag    
def avg_kurtosis(x, lag, bins, dt):
    '''Average kurtosis over spatial dimensions at given vector of lag times lag'''
    kurt = np.zeros(lag.shape)

    for j in range(len(lag)):
        tau = lag[j]*dt
        dx = (x[lag[j]:] - x[:-lag[j]])/tau  # Step (finite-difference derivative estimate)

        if x.ndim == 1:
            N = len(bins)
            for i in range(N):
                # Find where signal falls into this bin
                mask = (x[:-lag[j]] > bins[i]) * (x[:-lag[j]] < bins[i+1])
                mask_idx = np.nonzero(mask)[0]
                if len(mask_idx)>0:
                    kurt[j] = kurt[j] + kurtosis(dx[mask_idx])
        else: # 2D
            N = len(bins[0])*len(bins[1])
            id1 = np.digitize(x[:-lag[j],0],bins[0]) # which dimension 1 bin
            id2 = np.digitize(x[:-lag[j],1],bins[1]) # which dimension 2 bin
            uids = list(set(zip(id1,id2))) # unique bin ids
            if len(bins[0]) in id1 or len(bins[1]) in id2:
                raise ValueError('Data point outside of histogram bins')
            for uid in uids:
                mask = np.where((id1==uid[0])*(id2==uid[1]))[0]
                mask_idx = np.nonzero(mask)[0]
                if len(mask_idx)>0:
                    kurt[j] = kurt[j] + np.mean(kurtosis(dx[mask_idx]))
        kurt[j] = kurt[j]/N # Average over domain

    return kurt

# 1D Markov test

def ntrapz(I, dx):
    '''Numerical integration for arbitrary dimensions'''
    if isinstance(dx, int) or isinstance(dx, float) or len(dx)==1:
        return np.trapz(I, dx=dx, axis=-1)
    else:
        return np.trapz( ntrapz(I, dx[1:]), dx=dx[0])

def kl_divergence(p_in, q_in, dx=1, tol=None):
    """
    Approximate Kullback-Leibler divergence for arbitrary dimensions
    """
    if tol==None:
        tol = max( min(p_in.flatten()), min(q_in.flatten()))
    q = q_in.copy()
    p = p_in.copy()
    q[q<tol] = tol
    p[p<tol] = tol
    return ntrapz(p*np.log(p/q), dx)

def markov_test(X, lag, N=32, L=2):
    '''Test for Markov assumption in time series data: compares three-time PDF
    p(X3,t+lag|X2,t,X1,t-lag) with the Markovian p(X3,t+lag|X2,t)p(X2,t|X1,t-lag)
    via the KL divergence between the two PDFs.'''
    # Lagged time series
    X1 = X[:-2*lag:lag]
    X2 = X[lag:-lag:lag]
    X3 = X[2*lag::lag]
    
    # Two-time joint pdfs
    bins = np.linspace(-L, L, N+1)
    dx = bins[1]-bins[0]
    p12, _, _ = np.histogram2d(X1, X2, bins=[bins, bins], density=True)
    p23, _, _ = np.histogram2d(X2, X3, bins=[bins, bins], density=True)
    p2, _ = np.histogram(X2, bins=bins, density=True)
    p2[p2<1e-4] = 1e-4
    
    # Conditional PDF (Markov assumption)
    pcond_23 = p23.copy()
    for j in range(pcond_23.shape[1]):
        pcond_23[:, j] = pcond_23[:, j]/p2
        
    # Three-time PDFs
    p123, _ = np.histogramdd(np.array([X1, X2, X3]).T, bins=np.array([bins, bins, bins]), density=True)
    p123_markov = np.einsum('ij,jk->ijk',p12, pcond_23)
    
    # Chi^2 value
    #return utils.ntrapz( (p123 - p123_markov)**2, [dx, dx, dx] )/(np.var(p123.flatten()) + np.var(p123_markov.flatten()))
    return kl_divergence(p123, p123_markov, dx=[dx, dx, dx], tol=1e-6)