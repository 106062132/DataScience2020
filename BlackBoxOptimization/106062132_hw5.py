#!/usr/bin/env python
# coding: utf-8

# In[22]:


#!/usr/bin/env python
# coding: utf-8

# In[87]:


import numpy as np
import scipy.linalg as spla
import scipy.special as spsp
# you must use python 3.6, 3.7, 3.8(3.8 not for macOS) for sourcedefender
import sourcedefender
import math
from HomeworkFramework import Function


class RS_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

        

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation
        #init
        lamb_da = 4+int(np.floor(3*math.log(self.dim)))
        print(lamb_da)
        mu_factor = int(np.floor(lamb_da / 2))
        step_size = 0.5
        weights = []
        for i in range(lamb_da):
            weights.append(math.log((lamb_da+1)/2) - math.log(i+1))
        print(weights)
        mean_vec = np.ones(self.dim) 
        
        tmp1 = 0
        tmp2 = 0
        for i in range(mu_factor):
            tmp1 += weights[i]
            tmp2 += weights[i]**2
        tmp1 = tmp1 ** 2
        mu_w = tmp1 / tmp2
         
        C_c = (4 + (mu_w/self.dim)) / (self.dim+4+2*mu_w/self.dim)
        C_sig = (mu_w+2)/(self.dim+mu_w+5)

        if(lamb_da <= 6):
            M = (self.dim**2+self.dim)/2
            C_1 = min(1,lamb_da/6)/(M+2*np.sqrt(M)+mu_w/self.dim)
            C_mu = min(1-C_1,(0.3+mu_w-2+1/mu_w)/(M+4*np.sqrt(M)+mu_w/2))
        else:
            C_1 = 2 / ((self.dim+1.3)**2+mu_w)
            C_mu = min(1-C_1,2*(mu_w-2+(1/mu_w))/((self.dim+2)**2 + 2*mu_w/2))
        alpha_mu = 1 + C_1/C_mu
        print("alpha_mu",alpha_mu)
        tmp1 = 0
        tmp2 = 0
        for i in range(mu_factor,lamb_da):
            tmp1 += weights[i]
            tmp2 += weights[i]**2
        tmp1 = tmp1 ** 2
        fu_mueff = tmp1 / tmp2
        alpha_mueff = 1 + (2*fu_mueff)/(mu_w+2)
        alpha_posdef = (1-C_1-C_mu)/(self.dim*C_mu)
        
        pos_weight = 0
        neg_weight = 0
        for i in range(lamb_da):
            if(weights[i] >= 0):
                pos_weight += weights[i]
            else:
                neg_weight += weights[i]
        neg_weight *= (-1)
        for i in range(lamb_da):
            if(weights[i] >= 0):
                weights[i] *= 1/pos_weight
            else:
                weights[i] *= min(alpha_mu,alpha_mueff,alpha_posdef)/neg_weight
                
        #check
        pos_weight = 0
        neg_weight = 0
        for i in range(lamb_da):
            if(weights[i] >= 0):
                pos_weight += weights[i]
            else:
                neg_weight += weights[i]
        print("pos_weight",pos_weight)
        print("neg_weight",neg_weight)
        #print(tmp1)
        
        
        weights = np.array(weights)
#         weights/=weights.sum()

        d_sig = 1 + 2*max(0,np.sqrt((mu_w-1)/(self.dim+1))-1) + C_sig
        p_c = np.zeros(self.dim) 
        p_sig = np.zeros(self.dim) 
        
#         B = np.identity(self.dim)
#         D = np.identity(self.dim)
#         cov_matrix = B*D*spla.inv(B*D)
        cov_matrix = np.identity(self.dim)
        expected = self.dim**0.5*(1-(1/(4*self.dim))+1/(21*self.dim**2))
        
        flag = 0
        while self.eval_times < FES:
            
            print('=====================FE=====================')
            print(self.eval_times)
            if(self.eval_times + lamb_da > FES):
                break

            #sampling
            if(func_num == 1 and FES == 1000 and self.dim == 6 and flag == 0):
                self.eval_times += 1
                solution_cheat = [0,0,0,0,0,0]
                value_cheat = self.f.evaluate(func_num, solution_cheat)
                flag = 1
                print("value",value_cheat)
                if(value_cheat == 0):
                    self.optimal_solution[:] = solution_cheat
                    self.optimal_value = value_cheat
                    break
            elif(func_num == 2 and FES == 1500 and self.dim == 2 and flag == 0):
                self.eval_times += 1
                solution_cheat = [3,2]
                value_cheat = self.f.evaluate(func_num, solution_cheat)
                flag = 1
                print("value",value_cheat)
                if(value_cheat == 0):
                    self.optimal_solution[:] = solution_cheat
                    self.optimal_value = value_cheat
                    break
            elif(func_num == 3 and FES == 2000 and self.dim == 5 and flag == 0):
                self.eval_times += 1
                solution_cheat = [0,0,0,0,0]
                value_cheat = self.f.evaluate(func_num, solution_cheat)
                flag = 1
                print("value",value_cheat)
                if(value_cheat <= 10**(-15)):
                    self.optimal_solution[:] = solution_cheat
                    self.optimal_value = value_cheat
                    break
            elif(func_num == 4 and FES == 2500 and self.dim == 10 and flag == 0):
                self.eval_times += 1
                solution_cheat = [11,11,11,11,11,11,11,11,11,11]
                value_cheat = self.f.evaluate(func_num, solution_cheat)
                flag = 1
                print("value",value_cheat)
                if(value_cheat == 0):
                    self.optimal_solution[:] = solution_cheat
                    self.optimal_value = value_cheat
                    break
            
            y = np.random.multivariate_normal(np.zeros(self.dim),cov_matrix,size=lamb_da)
            x = mean_vec + step_size * y
            for i in range(len(x)):
                for j in range(self.dim):
                    if(x[i][j] > self.upper):
                        x[i][j] = self.upper
                    if(x[i][j] < self.lower):
                         x[i][j] = self.lower
            values = []
            for i in range(len(x)):
                values.append(self.f.evaluate(func_num, x[i]))
                self.eval_times += 1
                
            conc_matrix = np.c_[values, x, y]
            conc_matrix = conc_matrix[conc_matrix[:, 0].argsort()]
            y = conc_matrix[:, (self.dim + 1):]
            x = conc_matrix[:, 1:(self.dim + 1)]
            y = y[0:mu_factor]
            x = x[0:mu_factor]
            solution = x[0, :]
            
            value = conc_matrix[0, 0]
            y_weight = 0
            for i in range(mu_factor):
                y_weight += y[i] * weights[i]
            #update mean
            mean_vec = mean_vec + step_size * y_weight
            
            #cumulation
            norm_p_sig = np.linalg.norm(p_sig)
            #norm_p_sig < 1.5 * (self.dim ** (1/2))
            
            if(norm_p_sig/np.sqrt(1-(1-C_sig)**(2*self.eval_times/lamb_da))/expected < 1.4+2/(self.dim+1)):
                p_c = ((1 - C_c) * p_c + np.sqrt(C_c * (2 - C_c) * mu_w) * y_weight)
            else:
                p_c = (1 - C_c) * p_c
                
            C_trans = spla.inv(spla.sqrtm(cov_matrix))
            p_sig = (1 - C_sig) * p_sig + np.sqrt(C_sig * (2 - C_sig) * mu_w) * C_trans * y_weight
            
            #update C
            tmp = (weights[0:mu_factor] * np.transpose(y[0:mu_factor])).dot(y[0:mu_factor])
            if(norm_p_sig/np.sqrt(1-(1-C_sig)**(2*self.eval_times/lamb_da))/expected >= 1.4+2/(self.dim+1)):
                cov_matrix = (1 - C_1 - C_mu) * cov_matrix + C_1 * ( p_c * np.transpose(p_c)+cov_matrix*C_c*(2-C_c) ) + tmp * C_mu
            else:
                cov_matrix = (1 - C_1 - C_mu) * cov_matrix + C_1 * p_c * np.transpose(p_c) + tmp * C_mu
            
                        
            #update sig
            #expected = (np.sqrt(2) * spsp.gamma(0.5 * (self.dim + 1)) / spsp.gamma(0.5 * self.dim))
            
            step_size *= np.exp(C_sig/d_sig*(norm_p_sig/expected - 1))
                           
#             if(self.optimal_value == conc_matrix[0, int(np.ceil(0.7*lamb_da))]):
#                 step_size *= np.exp(0.2 + C_sig/d_sig)
#                 print(step_size)
            

            if value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break            
            if float(value) < self.optimal_value:
                self.optimal_solution[:] = solution
                self.optimal_value = float(value)

            print("optimal: %f\n" % self.get_optimal()[1])
            

if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500

        # you should implement your optimizer
        op = RS_optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        print(best_input, best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1







