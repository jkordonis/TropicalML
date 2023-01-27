import numpy as np


 


m_q=2
np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/m_q.npy',m_q)

L1_Err_mat=np.zeros(45)
Trop_div_Err_mat=np.zeros(45)

cnt=0
for i1 in range(10):
    for i2 in range(10):
        if i1<i2:
            print(m_q,i1,i2,cnt)
            np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/i1.npy',i1)
            np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/i2.npy',i2)


            runfile('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/F_Mnist_Compress_i1_i2.py', wdir='C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist')


            L1_err=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/L1_err.npy')
            Trop_div_error=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/Trop_div_error.npy')
            
            Trop_div_Err_mat[cnt]=Trop_div_error
            L1_Err_mat[cnt]=L1_err
            cnt=cnt+1
            

Trop_div_2_mean = Trop_div_Err_mat.mean()
Trop_div_2_std = Trop_div_Err_mat.std()

L1_Err_2_mean=L1_Err_mat.mean()
L1_Err_2_std=L1_Err_mat.std()


# m_q = 3 -------------------------------------------------------------------------

m_q=3
np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/m_q.npy',m_q)

L1_Err_mat=np.zeros(45)
Trop_div_Err_mat=np.zeros(45)

cnt=0
for i1 in range(10):
    for i2 in range(10):
        if i1<i2:
            print(m_q,i1,i2,cnt)
            np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/i1.npy',i1)
            np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/i2.npy',i2)


            runfile('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/F_Mnist_Compress_i1_i2.py', wdir='C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist')


            L1_err=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/L1_err.npy')
            Trop_div_error=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/Trop_div_error.npy')
            
            Trop_div_Err_mat[cnt]=Trop_div_error
            L1_Err_mat[cnt]=L1_err
            cnt=cnt+1
            

Trop_div_3_mean = Trop_div_Err_mat.mean()
Trop_div_3_std = Trop_div_Err_mat.std()

L1_Err_3_mean=L1_Err_mat.mean()
L1_Err_3_std=L1_Err_mat.std() 


# m_q = 4 -------------------------------------------------------------------------

m_q=4
np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/m_q.npy',m_q)

L1_Err_mat=np.zeros(45)
Trop_div_Err_mat=np.zeros(45)

cnt=0
for i1 in range(10):
    for i2 in range(10):
        if i1<i2:
            print(m_q,i1,i2,cnt)
            np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/i1.npy',i1)
            np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/i2.npy',i2)


            runfile('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/F_Mnist_Compress_i1_i2.py', wdir='C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist')


            L1_err=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/L1_err.npy')
            Trop_div_error=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/Trop_div_error.npy')
            
            Trop_div_Err_mat[cnt]=Trop_div_error
            L1_Err_mat[cnt]=L1_err
            cnt=cnt+1
            

Trop_div_4_mean = Trop_div_Err_mat.mean()
Trop_div_4_std = Trop_div_Err_mat.std()

L1_Err_4_mean=L1_Err_mat.mean()
L1_Err_4_std=L1_Err_mat.std() 


# m_q = 5 -------------------------------------------------------------------------

m_q=5
np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/m_q.npy',m_q)

L1_Err_mat=np.zeros(45)
Trop_div_Err_mat=np.zeros(45)

cnt=0
for i1 in range(10):
    for i2 in range(10):
        if i1<i2:
            print(m_q,i1,i2,cnt)
            np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/i1.npy',i1)
            np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/i2.npy',i2)


            runfile('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/F_Mnist_Compress_i1_i2.py', wdir='C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist')


            L1_err=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/L1_err.npy')
            Trop_div_error=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/Trop_div_error.npy')
            
            Trop_div_Err_mat[cnt]=Trop_div_error
            L1_Err_mat[cnt]=L1_err
            cnt=cnt+1
            

Trop_div_5_mean = Trop_div_Err_mat.mean()
Trop_div_5_std = Trop_div_Err_mat.std()

L1_Err_5_mean=L1_Err_mat.mean()
L1_Err_5_std=L1_Err_mat.std() 


# m_q = 6 -------------------------------------------------------------------------

m_q=6
np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/m_q.npy',m_q)

L1_Err_mat=np.zeros(45)
Trop_div_Err_mat=np.zeros(45)

cnt=0
for i1 in range(10):
    for i2 in range(10):
        if i1<i2:
            print(m_q,i1,i2,cnt)
            np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/i1.npy',i1)
            np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/i2.npy',i2)


            runfile('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/F_Mnist_Compress_i1_i2.py', wdir='C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist')


            L1_err=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/L1_err.npy')
            Trop_div_error=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/Trop_div_error.npy')
            
            Trop_div_Err_mat[cnt]=Trop_div_error
            L1_Err_mat[cnt]=L1_err
            cnt=cnt+1
            

Trop_div_6_mean = Trop_div_Err_mat.mean()
Trop_div_6_std = Trop_div_Err_mat.std()

L1_Err_6_mean=L1_Err_mat.mean()
L1_Err_6_std=L1_Err_mat.std() 



# m_q = 7 -------------------------------------------------------------------------

m_q=7
np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/m_q.npy',m_q)

L1_Err_mat=np.zeros(45)
Trop_div_Err_mat=np.zeros(45)

cnt=0
for i1 in range(10):
    for i2 in range(10):
        if i1<i2:
            print(m_q,i1,i2,cnt)
            np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/i1.npy',i1)
            np.save('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/i2.npy',i2)


            runfile('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/F_Mnist_Compress_i1_i2.py', wdir='C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist')


            L1_err=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/L1_err.npy')
            Trop_div_error=np.load('C:/TensorFlowExamples/runn2/TensorFlowExamples_transf/fashion_mnist/Trop_div_error.npy')
            
            Trop_div_Err_mat[cnt]=Trop_div_error
            L1_Err_mat[cnt]=L1_err
            cnt=cnt+1
            

Trop_div_7_mean = Trop_div_Err_mat.mean()
Trop_div_7_std = Trop_div_Err_mat.std()

L1_Err_7_mean=L1_Err_mat.mean()
L1_Err_7_std=L1_Err_mat.std() 







L1_mean=[L1_Err_2_mean,L1_Err_3_mean,L1_Err_4_mean,L1_Err_5_mean,L1_Err_6_mean,L1_Err_7_mean]
Trop_div_mean=[Trop_div_2_mean,Trop_div_3_mean,Trop_div_4_mean,Trop_div_5_mean,Trop_div_6_mean,Trop_div_7_mean]

L1_std=[L1_Err_2_std,L1_Err_3_std,L1_Err_4_std,L1_Err_5_std,L1_Err_6_std,L1_Err_7_std]
Trop_div_std=[Trop_div_2_std,Trop_div_3_std,Trop_div_4_std,Trop_div_5_std,Trop_div_6_std,Trop_div_7_std]





