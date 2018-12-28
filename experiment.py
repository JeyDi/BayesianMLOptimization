# import of the required libraries
import numpy as np
import timeit


from pyGPGO.covfunc import squaredExponential
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.surrogates.RandomForest import RandomForest
from pyGPGO.GPGO import GPGO
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import matern32

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show


#Import Load Wine Dataset
ds = datasets.load_wine()
print("Dataframe Data Shape: ",ds.data.shape)
print("Dataframe Target Shape: ", ds.target.shape)

def compute_accuracy_SVC(C,gamma,coef0):
        clf = svm.SVC(C=C,gamma=gamma,coef0=coef0)
        scores = cross_val_score(clf, ds.data, ds.target, cv=10)
        return (scores.mean())

np.random.seed(42)
initialPoints = 30
furtherEvaluations = 120

# defining a dictionary on "x"
param = {   'C': ('cont', [0.1,5]),
            'gamma': ('cont', [0.1,10]),
            'coef0':('cont',[0.1,10])
        } 

# creating a GP surrogate model with a Squared Exponantial covariance function,
# aka kernel
sexp = squaredExponential()
sur_model_1 = GaussianProcess(sexp)
sur_model_2 = RandomForest()

# setting the acquisition function
acq_1 = Acquisition(mode="ExpectedImprovement")
acq_2 = Acquisition(mode="ProbabilityImprovement")
acq_3 = Acquisition(mode="UCB")

# creating an object Bayesian Optimization
gpgo_gaussian_model_1 = GPGO(sur_model_1,acq_1,compute_accuracy_SVC,param)
gpgo_gaussian_model_2 = GPGO(sur_model_1,acq_2,compute_accuracy_SVC,param)
gpgo_gaussian_model_3 = GPGO(sur_model_1,acq_3,compute_accuracy_SVC,param)

gpgo_random_forest_1 = GPGO(sur_model_2,acq_1,compute_accuracy_SVC,param)
gpgo_random_forest_2 = GPGO(sur_model_2,acq_2,compute_accuracy_SVC,param)
gpgo_random_forest_3 = GPGO(sur_model_2,acq_3,compute_accuracy_SVC,param)

#Run models
gaussianModel_1_start = timeit.default_timer()
gpgo_gaussian_model_1.run(max_iter=furtherEvaluations,init_evals=initialPoints)
gaussianModel_1_stop = timeit.default_timer()

gaussianModel_2_start = timeit.default_timer()
gpgo_gaussian_model_2.run(max_iter=furtherEvaluations,init_evals=initialPoints)
gaussianModel_2_stop = timeit.default_timer()

gaussianModel_3_start = timeit.default_timer()
gpgo_gaussian_model_3.run(max_iter=furtherEvaluations,init_evals=initialPoints)
gaussianModel_3_stop = timeit.default_timer()

randomForest_1_start = timeit.default_timer()
gpgo_random_forest_1.run(max_iter=furtherEvaluations,init_evals=initialPoints)
randomForest_1_stop = timeit.default_timer()

randomForest_2_start = timeit.default_timer()
gpgo_random_forest_2.run(max_iter=furtherEvaluations,init_evals=initialPoints)
randomForest_2_stop = timeit.default_timer()

randomForest_3_start = timeit.default_timer()
gpgo_random_forest_3.run(max_iter=furtherEvaluations,init_evals=initialPoints)
randomForest_3_stop = timeit.default_timer()

#Get the results
print("\n---Results---\n")
print("\n", gpgo_gaussian_model_1.getResult())
print('Gaussian Model 1 Time: ', gaussianModel_1_start - gaussianModel_1_stop)  
print("\n", gpgo_gaussian_model_2.getResult())
print('Gaussian Model 2 Time: ', gaussianModel_2_start - gaussianModel_2_stop)  
print("\n", gpgo_gaussian_model_3.getResult())
print('Gaussian Model 3 Time: ', gaussianModel_3_start - gaussianModel_3_start)  

print("\n", gpgo_random_forest_1.getResult())
print('Random Forest 1 Time: ', randomForest_1_start - randomForest_1_stop)  
print("\n", gpgo_random_forest_2.getResult())
print('Random Forest 2 Time: ', randomForest_2_start - randomForest_2_stop)  
print("\n",gpgo_random_forest_3.getResult())
print('Random Forest 3 Time: ', randomForest_3_start - randomForest_3_stop)  


#------------------------------
#GRID SEARCH
xrange = list(param.values())[0][1]
yrange = list(param.values())[1][1]
zrange = list(param.values())[2][1]


C_values = np.linspace(xrange[0],xrange[1],5)
gamma_values = np.linspace(yrange[0],yrange[1],6)
def0 = np.linspace(zrange[0],zrange[1],5)

res = [0 for n in range(150)]
count = 0
grid_search_start = timeit.default_timer()
for cc in C_values:
    for gg in gamma_values:
        for dd in def0:
            res[count] = compute_accuracy_SVC( cc, gg, dd )
            count = count+1

grid_search_stop = timeit.default_timer()

print("\nGrid Search, Best on Grid:"+str(round(max(np.asarray(res)),2))+"%%")
print('Grid Search Time: ', grid_search_start - grid_search_stop)  
       


print("\n\n---Finish---")