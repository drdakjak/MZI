
# coding: utf-8

# In[5]:

# Zdroj: http://mi21.vsb.cz/sites/mi21.vsb.cz/files/unit/linearni_algebra_s_matlabem.pdf


# In[1]:

import sys, getopt
import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp
import pandas as pd


# In[2]:

def main(argv):
    inputfile = ''
    r = ''    
    try:
        opts, args = getopt.getopt(argv,"hfdi:",["ifile="])
    except getopt.GetoptError:
        print( 'qr.py {-f (full);-d (dict)} -i <inputfile>')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print( 'qr.py {-f (full);-d (dict)} -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt == "-f":
            r = "f"
        elif opt == "-d":
            r = "d"
            
    print( 'Input file is ', inputfile)
    
    Q, R = QR_decomp(inputfile, r)


# In[3]:

def QR_decomp(inputfile, opt):
    data = pd.read_csv(inputfile,delimiter=';',header=None)
    if(opt=='f'):
#         display(data)
        A = sp.lil_matrix(data)
    elif(opt=='d'):
#         display(data)
        row = data[0]
        col = data[1]
        data = data[2]
        A = sp.coo_matrix((data, (row, col))).tolil()
        
    
    m = A.shape[0]
    n = A.shape[1]
    Q = sp.eye(m = m).tolil()
    R=A
    for j in range(n):
        for i in range(m-1, j, -1):
            x = R[:,j]
            if(norm([x[i-1],x[i]]) > 0):
                c = x[i-1]/norm([x[i-1],x[i]])
                s = -x[i]/norm([x[i-1],x[i]])
                G = sp.eye(m = m).tolil()
                G[i-1,i-1:i+1] = [round(c.item(),8), round(s.item(),8)]
                G[i,i-1:i+1] = [round(-s.item(),8),round(c.item(),8)]
                R = G.T.dot(R)
                Q = Q.dot(G)
    if(opt=='f'):
        Q, R = np.nan_to_num(Q.todense()), np.nan_to_num(R.todense()).round(8)
        np.savetxt("Q_f.csv", Q, delimiter=";")
        np.savetxt("R_f.csv", R, delimiter=";")
        return Q, R

    if(opt=='d'):
        Q = sp.coo_matrix(Q)
        R.data = R.data.round(8)
        R.eliminate_zeros()
        R = sp.coo_matrix(R)
        pd.DataFrame([Q.row, Q.col, Q.data]).T.to_csv("Q_d.csv",sep=";",header=False, index= False)
        pd.DataFrame([R.row, R.col, R.data]).T.to_csv("R_d.csv",sep=";",header=False, index= False)

        return Q, R


# In[4]:

if __name__ == "__main__":
    main(sys.argv[1:])


# In[76]:

# sp.coo_matrix(A).ceil().todense()
# A = np.random.rand(50*100).reshape(50,100)
# np.savetxt("M.csv", A, delimiter=";")
# pd.DataFrame([sp.coo_matrix(A).row, sp.coo_matrix(A).col, sp.coo_matrix(A).data]).T.to_csv("M_sparse.csv",sep=";",header=False, index= False)


# In[77]:




# In[78]:




# In[79]:




# In[80]:




# In[ ]:



