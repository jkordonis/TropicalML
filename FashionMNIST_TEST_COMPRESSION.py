# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:46:34 2022

@author: jkord
"""

import tensorflow as tf
import numpy as np
from scipy.linalg import block_diag


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

model.evaluate(test_images,test_labels)



W1=model.layers[1].weights[0].numpy()
b1=model.layers[1].weights[1].numpy()
W2=model.layers[2].weights[0].numpy()
b2=model.layers[2].weights[1].numpy()
 


x_train=train_images
y_train=train_labels
x_test=test_images
y_test=test_labels
Ner=128



x_test_vec =np.zeros([10000, 784])
for i in range(10000):
  b = np.reshape(x_test[i],(784))
  x_test_vec[i]=b


# Parameters
Tot_results=np.zeros((30,3))
RESU=np.zeros((6,6))

ii1=np.zeros(30)
ii2=np.zeros(30)

for  TEST_num in range(30):
    i1=np.random.randint(10)
    i2=np.random.randint(10)
    if i1==i2:
        i2=np.random.randint(10)
    if i1==i2:
        i2=np.random.randint(10)
    if i1==i2:
        i2=np.random.randint(10)
    if i1==i2:
        i2=np.random.randint(10)
    ii1[TEST_num]=i1
    ii2[TEST_num]=i2



NUMB_OF_TERMS=[3,5,10]
Tot_results=np.zeros((30,3))
for numb_of_terms_cnt in range(3):
    m_q=NUMB_OF_TERMS[numb_of_terms_cnt]
    for TEST_num in range(30):
        print(numb_of_terms_cnt ,TEST_num )
        i1=int(ii1[TEST_num])
        i2=int(ii2[TEST_num])
        print('comparing  ',class_names[i1],' and ' ,class_names[i2])
        
        
        x_test_i1_i2=[]
        y_test_i1_i2=[]
        
        
        for i in range(10000):
          if y_test[i]==i1 or y_test[i]==i2:
            x_test_i1_i2.append(x_test_vec[i])
            y_test_i1_i2.append(-1)
            if y_test[i]==i1:
              y_test_i1_i2[-1]=1
            if y_test[i]==i2:
              y_test_i1_i2[-1]=0
        
        x_test_i1_i2=np.array(x_test_i1_i2)
        y_test_i1_i2=np.array(y_test_i1_i2)
        
        
        x_train_i1_i2=[]
        y_train_i1_i2=[]
        x_train_vec =np.zeros([60000, 784])
        for i in range(60000):
          b = np.reshape(x_train[i],(784))
          x_train_vec[i]=b
        
        
        
        for i in range(60000):
           if y_train[i]==i1 or y_train[i]==i2:
            x_train_i1_i2.append(x_train_vec[i])
            y_train_i1_i2.append(-1)
            if y_train[i]==i1:
              y_train_i1_i2[-1]=1
            if y_train[i]==i2:
              y_train_i1_i2[-1]=0
        
        x_train_i1_i2=np.array(x_train_i1_i2)
        y_train_i1_i2=np.array(y_train_i1_i2)
        
        err=0
        for i in range(x_test_i1_i2.shape[0]):
          z =  np.matmul(W1.T,x_test_i1_i2[i]) +b1
          z=np.maximum(z, np.zeros(Ner))
          y=np.matmul(W2.T,z)+b2
          err_i=0
          if y[i1]>y[i2] and y_test_i1_i2[i]==0:
            err_i=1
          if y[i1]<y[i2] and y_test_i1_i2[i]==1:
            err_i=1
          err=err+err_i
        
        TST_ERR_ORIG = err/x_test_i1_i2.shape[0]
        
        
        #
        
        #  Dimensionality Reduction
        
        # Dimension Reduction
        W2_=W2[:,i1]-W2[:,i2]
        b2_=b2[i1]-b2[i2]
        
        err=0
        for i in range(x_test_i1_i2.shape[0]):
          z =  np.matmul(W1.T,x_test_i1_i2[i]) +b1
          z=np.maximum(z, np.zeros(Ner))
          y=np.matmul(W2_.T,z)+b2_
          err_i=0
          if y>0 and y_test_i1_i2[i]==0:#if y[3]>y[5] and y_test_i1_i2[i]==0:
            err_i=1
          if y<0 and y_test_i1_i2[i]==1:#if y[3]<y[5] and y_test_i1_i2[i]==1:
            err_i=1
          err=err+err_i
        err/x_test_i1_i2.shape[0]
        
        
        W2_pl=np.maximum(W2_,np.zeros(W2_.shape))
        W2_min=np.maximum(-W2_,np.zeros(W2_.shape))
        ap_1 = np.zeros((W1.shape[1],W1.shape[0]))
        bp_1 = np.zeros(W1.shape[1])
        ap_2 = np.zeros((W1.shape[1],W1.shape[0]))
        bp_2 = np.zeros(W1.shape[1])
        
        
        for i in range(W1.shape[1]):
          ap_1[i]=W2_pl[i]*W1.T[i]
          bp_1[i]=W2_pl[i]*b1[i]
          ap_2[i]=W2_min[i]*W1.T[i]
          bp_2[i]=W2_min[i]*b1[i]
        
        a1_vec=ap_1.T
        a2_vec=ap_2.T
        
        True_Val_1=(np.array([np.linalg.norm(a1_vec[:,i]) for i in range(a1_vec.shape[1])])!=0)
        True_Val_2=(np.array([np.linalg.norm(a2_vec[:,i]) for i in range(a2_vec.shape[1])])!=0)
        
        a1_nonz_vec  = a1_vec[:,True_Val_1]
        a2_nonz_vec  = a2_vec[:,True_Val_2]
        
        
        Q1,R1=np.linalg.qr(a1_nonz_vec,mode='reduced')
        Q2,R2=np.linalg.qr(a2_nonz_vec,mode='reduced')
         
        a1_red_vec = np.matmul(Q1.T,a1_nonz_vec)
        a2_red_vec = np.matmul(Q2.T,a2_nonz_vec)
        bp_1_red=bp_1[True_Val_1]
        bp_2_red=bp_2[True_Val_2]
        
        
        
        X_sample1 = np.matmul(x_train_i1_i2,Q1) 
        X_sample2 = np.matmul(x_train_i1_i2,Q2) 
        
        
        X_sample1_test = np.matmul(x_test_i1_i2,Q1) 
        X_sample2_test = np.matmul(x_test_i1_i2,Q2) 
        
        
        
        
        
        # Functionsss 
        def tropical_sum_pol_function(x,pol):
          a_=pol[0]
          b_=pol[1]
          s= 0
          for i in range(np.shape(a_)[0]):
              s=s+max(np.inner(a_[i],x)+b_[i],0)
          return(s)
        
        # Phase 1
        def Phase_1_function_comp_quotient(X_sample,q_pol):
          a_hat,b_hat=q_pol
          m_q=a_hat.shape[0]
          # Initialize sets I_i
          c1 = np.zeros(X_sample.shape[1]*m_q)
          y=np.zeros((m_q,X_sample.shape[0]))
          for i in range(y.shape[0]):
            for j in range(y.shape[1]):
              y[i,j] = 1 if np.inner(a_hat[i],X_sample[j])+b_hat[i]>=0 else 0
          c2=np.sum(y,1)
          c2=np.sum(y,1)
          c11=np.matmul(y,X_sample)
          c1=c11.reshape(c11.shape[0]*c11.shape[1])
          c1.shape
          return c1,c2
        
        def Phase_1_function_comp_quotient_z(X_sample,q_pol,A,B):
          a_hat,b_hat=q_pol
          m_q=a_hat.shape[0]
          d=X_sample.shape[1]
          # Initialize sets I_i
          y=np.zeros((m_q,X_sample.shape[0]))
          for i in range(m_q):
            for j in range(X_sample.shape[0]):
              y[i,j] = 1 if np.inner(a_hat[i],X_sample[j])+b_hat[i]>=0 else 0
          c2=np.sum(y,1)
          c11=np.matmul(y,X_sample)
          c_til=np.zeros((m_q,d))
          for i in range(m_q):
            c_til[i] = c11[i]@A+c2[i]*B 
          return c_til
        
        
        
        
        
        # Division 1
        
        
        # Martix Definitions
        
        A=a1_red_vec
        d=A.shape[1]
        m_p=A.shape[0]
        B=bp_1_red
        
        p_pol=(A.T,B)
        
        
        A_inverse = np.linalg.inv(A)
        B_transp_A_inverse = np.matmul(B.T,A_inverse)
        B_transp_A_inverse_mat=B_transp_A_inverse 
        A_inverse_hor_conc=A_inverse
        
        
        
        a_hat, b_hat=np.array(np.random.randn(m_q,d)), np.array(np.random.randn(m_q))
        q_pol=(a_hat, b_hat)
        X_sample=X_sample1[0:200]
        N_sample=X_sample1.shape[0]
        
        A_inv_mat = A_inverse
        
        for i in range(m_q-1):
          A_inv_mat=block_diag(A_inv_mat,A_inverse) 
          B_transp_A_inverse_mat=block_diag(B_transp_A_inverse_mat,B_transp_A_inverse)
          A_inverse_hor_conc=np.concatenate((A_inverse_hor_conc,A_inverse),axis=1)
        
        
        
        f_x_i=np.zeros(np.shape(X_sample)[0])
        for i in range(np.shape(X_sample)[0]):
          f_x_i[i]=tropical_sum_pol_function(X_sample[i],p_pol)
        
        
        c1,c2=Phase_1_function_comp_quotient(X_sample,q_pol)
        c2
        
        
        
        
        EPAN=50
        Iterations=15
        
        A_inv=np.linalg.inv(A)
        progress_mat=np.zeros(Iterations+1)
        current_err=1e10
        a_new=np.zeros(a_hat.shape)
        b_new=np.zeros(b_hat.shape)
        
        
        for ii in range(EPAN):
          progress_mat[0]=np.sum(f_x_i)-np.sum([tropical_sum_pol_function(X_sample[i],q_pol) for i in range(X_sample.shape[0])])
          a_hat, b_hat=0.01*np.array(np.random.randn(m_q,d)), 0.01*np.array(np.random.randn(m_q))
          for iter in range(Iterations):
            q_pol = (a_hat,b_hat)
            c_til=Phase_1_function_comp_quotient_z(X_sample,q_pol,A,B)
            z=np.zeros((m_q,d))
            indd =np.argmax(c_til, axis = 0)
            for l in range(d):
              if c_til[indd[l],l]>0:
                z[indd[l],l]=1
            for i in range(m_q):
              a_new[i]= A@z[i]
              b_new[i]= B@z[i]
            a_hat=a_hat*0.01+0.99*a_new
            b_hat=b_hat*0.01+0.99*b_new
            q_pol = (a_hat,b_hat)
            progress_mat[iter+1]=np.sum(f_x_i)-np.sum([tropical_sum_pol_function(X_sample[i],q_pol) for i in range(X_sample.shape[0])])
            if progress_mat[iter+1]<current_err:
              current_err=progress_mat[iter+1]
              a_hat_1 = np.ndarray.copy(a_hat)
              b_hat_1 = np.ndarray.copy(b_hat)
        
        # Division 2
        
        # Martix Definitions
        
        A=a2_red_vec
        d=A.shape[1]
        m_p=A.shape[0]
        B=bp_2_red
        p_pol=(A.T,B)
        
        
        A_inverse = np.linalg.inv(A)
        B_transp_A_inverse = np.matmul(B.T,A_inverse)
        B_transp_A_inverse_mat=B_transp_A_inverse 
        A_inverse_hor_conc=A_inverse
        
        
        
        a_hat, b_hat=np.array(np.random.randn(m_q,d)), np.array(np.random.randn(m_q))
        q_pol=(a_hat, b_hat)
        X_sample=X_sample2[0:200]
        N_sample=X_sample2.shape[0]
        
        A_inv_mat = A_inverse
        
        for i in range(m_q-1):
          A_inv_mat=block_diag(A_inv_mat,A_inverse) 
          B_transp_A_inverse_mat=block_diag(B_transp_A_inverse_mat,B_transp_A_inverse)
          A_inverse_hor_conc=np.concatenate((A_inverse_hor_conc,A_inverse),axis=1)
        
        
        f_x_i=np.zeros(np.shape(X_sample)[0])
        for i in range(np.shape(X_sample)[0]):
          f_x_i[i]=tropical_sum_pol_function(X_sample[i],p_pol)
        
        
        #TESTING
        EPAN=50
        
        A_inv=np.linalg.inv(A)
        progress_mat=np.zeros(Iterations+1)
        current_err=1e10
        a_new=np.zeros(a_hat.shape)
        b_new=np.zeros(b_hat.shape)
        
        
        for ii in range(EPAN):
          progress_mat[0]=np.sum(f_x_i)-np.sum([tropical_sum_pol_function(X_sample[i],q_pol) for i in range(X_sample.shape[0])])
          a_hat, b_hat=0.01*np.array(np.random.randn(m_q,d)), 0.01*np.array(np.random.randn(m_q))
          for iter in range(Iterations):
            q_pol = (a_hat,b_hat)
            c_til=Phase_1_function_comp_quotient_z(X_sample,q_pol,A,B)
            z=np.zeros((m_q,d))
            indd =np.argmax(c_til, axis = 0)
            for l in range(d):
              if c_til[indd[l],l]>0:
                z[indd[l],l]=1
            for i in range(m_q):
              a_new[i]= A@z[i]
              b_new[i]= B@z[i]
            a_hat=a_hat*0.01+0.99*a_new
            b_hat=b_hat*0.01+0.99*b_new
            q_pol = (a_hat,b_hat)
            progress_mat[iter+1]=np.sum(f_x_i)-np.sum([tropical_sum_pol_function(X_sample[i],q_pol) for i in range(X_sample.shape[0])])
            if progress_mat[iter+1]<current_err:
              current_err=progress_mat[iter+1]
              a_hat_2 = np.ndarray.copy(a_hat)
              b_hat_2 = np.ndarray.copy(b_hat)
        
        # Check
        
        err=0
        
        for i in range(X_sample1_test.shape[0]):
          p1_val = np.sum( np.maximum(np.matmul(a_hat_1,X_sample1_test[i]) +b_hat_1,np.zeros(a_hat_1.shape[0])))
          p2_val = np.sum( np.maximum(np.matmul(a_hat_2,X_sample2_test[i]) +b_hat_2,np.zeros(a_hat_2.shape[0])))
          p_val =p1_val-p2_val+b2_
          y=p_val
          err_i=0
          if y>0 and y_test_i1_i2[i]==0:#if y[3]>y[5] and y_test_i1_i2[i]==0:
            err_i=1
          if y<0 and y_test_i1_i2[i]==1:#if y[3]<y[5] and y_test_i1_i2[i]==1:
            err_i=1
          err=err+err_i
        ERRTr=err/X_sample1_test.shape[0]
        
        
        
        # L1 Structured Pruning
        
        A_diff=(ap_1-ap_2)
        normss = np.array([np.linalg.norm(A_diff[i],1) for i in range(100)])
        normss
        
        Indices_max =np.argsort(-normss)
        
        
        N_L1_red=2*m_q
        ap_1_L1_red = ap_1[Indices_max[0:N_L1_red]]
        ap_2_L1_red = ap_2[Indices_max[0:N_L1_red]]
        bp_1_L1_red = bp_1[Indices_max[0:N_L1_red]]
        bp_2_L1_red = bp_2[Indices_max[0:N_L1_red]]
        
        
        
        
        err=0
        for i in range(x_test_i1_i2.shape[0]):
          xx=x_test_i1_i2[i]
          y=np.sum(np.maximum(ap_1_L1_red@xx+bp_1_L1_red,0))-np.sum(np.maximum(ap_2_L1_red@xx+bp_2_L1_red,0))+b2_
          err_i=0
          if y>0 and y_test_i1_i2[i]==0:
            err_i=1
          if y<0 and y_test_i1_i2[i]==1:
            err_i=1
          err=err+err_i
          Tot_results[TEST_num,0]=TST_ERR_ORIG
          Tot_results[TEST_num,1]=ERRTr
          Tot_results[TEST_num,2]=err/x_test_i1_i2.shape[0]
          #Tot_results[TEST_num,1]=Tot_results[TEST_num,1]-Tot_results[TEST_num,0]
          #Tot_results[TEST_num,2]=Tot_results[TEST_num,2]-Tot_results[TEST_num,0]
          
        print(Tot_results[TEST_num])
    mean_RESU=np.mean(Tot_results,0)
    std_RESU = np.std(Tot_results,0)
    
    
    RESU[numb_of_terms_cnt]=np.array(np.concatenate((mean_RESU,std_RESU)))



        
        
        
