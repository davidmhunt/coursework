import numpy as np
from scipy.linalg import null_space


# General function to compute expected hitting time for Exercise 1
def compute_Phi_ET(P, ns=100):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain
        ns {int} -- largest step to consider

    Returns:
        Phi_list {numpy.array} -- (ns + 1) x n x n, the Phi matrix for time 0, 1, ...,ns
        ET {numpy.array} -- n x n, expected hitting time approximated by ns steps ns
    '''

    # Add code here to compute following quantities:
    
    #determine size of P
    n = P.shape[0]
    
    #pre-allocate matrix of (1 - I)
    one_minus_I = np.subtract(np.ones((n,n)),np.identity(n))
    
    #preallocate variables
    Phi_list = np.zeros((ns + 1,n,n))
    ET = np.zeros((n,n))
    
    #compute Phi_list and ET
    for i in range(0,ns+1):
        if i == 0:
            Phi_list[i,:,:] = np.array(np.identity(n))
            ET = np.zeros((n,n))
        else:
            
            #compute the next values
            
            # Phi_list[m, i, j] = phi_{i,j}^{(m)} = Pr( T_{i, j} <= m )
            Phi_list[i] = np.identity(n) + np.multiply(one_minus_I,np.matmul(P,Phi_list[i-1]))
            
            # ET[i, j] = E[ T_{i, j} ] ~ \sum_{m=1}^ns m Pr( T_{i, j} = m )
            Pr_i = np.subtract(Phi_list[i],Phi_list[i-1])
            ET = np.add(ET,i * Pr_i)
    
    # Notice in python the index starts from 0

    return Phi_list, ET


# General function to simulate hitting time for Exercise 1
def simulate_hitting_time(P, states, nr):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain
        states {list[int]} -- the list [start state, end state], index starts from 0
        nr {int} -- largest step to consider

    Returns:
        T {list[int]} -- a size nr list contains the hitting time of all realizations
    '''
    n = P.shape[0]
    T = [0] * nr
    
    for i in range(0,nr,1):
        hit_count = 0
        current_state = states[0]
        
        while ((current_state != states[1])):
            hit_count += 1
            current_state = np.random.choice(np.arange(n),p = P[current_state,:])

        T[i] = hit_count
    
    #run the simulation
    # Add code here to simulate following quantities:
    # T[i] = hitting time of the i-th run (i.e., realization) of process
    # Notice in python the index starts from 0
    
    return T



# General function to compute the stationary distribution (if unique) of a Markov chain for Exercise 3
def stationary_distribution(P):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain

    Returns:
        pi {numpy.array} -- length n, stationary distribution of the Markov chain
    '''

    # Add code here: Think of pi as column vector, solve linear equations:
    #     P^T pi = pi
    n = P.shape[0]
    pi = null_space(np.transpose(np.identity(n) - P))
    #pi = np.linalg.solve(np.transpose(np.identity(n) - P.transpose()),np.zeros((n,1)))
    #     sum(pi) = 1
    pi = pi / sum(pi)

    return pi[:,0]
