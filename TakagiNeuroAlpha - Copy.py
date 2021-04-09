import simpful as sf
import ANFIS as TK
import numpy as np
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt

#ts = np.loadtxt("data2.txt", usecols=[0,1,2,3])
ts = np.loadtxt("network.txt",usecols=[0,1,2,3])
#print(np.absolute(ts[0+1,3]))


# build a antecedent from bad to good. 
# if_Statement = [["long","average","short"],["high","medium","low"],\
          # ["low","medium","high"]]
# x_premis = [[.5,.5,.5],[.15,.15,.15],[2,3,2]]
# z_premis = [[2.5,1.5,0.5],[0.5,0.3,0.1],[2,4,7]]
# var =["delay","packet loss ratio","route load capacity","grade"]
# conseq = ["bad","poor","fair","average", "good", "excellent"]

if_Statement = [["weak","average","strong"],\
				["long","average","short"],\
					["bad","average","good"]]
sd_z = [[0.35,0.51,0.63], [0.38,0.31,0.32],[0.17,0.47,0.55]]
mean_x = [[5.01,5.94,6.59],[3.42,2.77,2.97],[1.46,4.26,5.55]]
var= ["R","T","B","grade"]
conseq = ["poor","bad","average","fair","good","excellent"]
yOut= ts[0,3]
z_conseq = [0.5,0.5,0.5,0.5,0.5,0.5]
	   
AdaptiveFIS = TK.Anfis(np.absolute(ts[0,0:3]),sd_z,mean_x,if_Statement,var,conseq\
					   ,yOutput=yOut, learningRate=0.01, memFuncType=[2,1,3],\
						   z=z_conseq)
AdaptiveFIS.If_Then()
membershipOut = AdaptiveFIS.L2()
AdaptiveFIS.L3()
AdaptiveFIS.L4(text = True)
Output = AdaptiveFIS.Defuzzication()
AdaptiveFIS.MSE(text=False)
[mean, sd] = AdaptiveFIS.BackPass()

[inputs, sd, mean, consequencesPara, y_out, y_pred,dE]=AdaptiveFIS.ObtainParas(True)

epochMSE = []
for y in range(300):
	mseLoop = []
	for x in range(148):
		 yOut = ts[x+1,3]
		 membershipOut=AdaptiveFIS.L2(ts[x+1,0:3])
		 AdaptiveFIS.L3()
		 AdaptiveFIS.L4()
		 Output = AdaptiveFIS.Defuzzication()
		 mse = AdaptiveFIS.MSE(yOutput=yOut)
		 mseLoop.append(mse)
	epochMSE.append(sum(mseLoop))		 

[inputs, sd, mean, consequencesPara, y_out, y_pred,dE]=AdaptiveFIS.ObtainParas(True)

dataPred = []
for x in range(148):
	membershipOut=AdaptiveFIS.L2(ts[x+1,0:3])
	AdaptiveFIS.L3()
	AdaptiveFIS.L4()
	Output = AdaptiveFIS.Defuzzication()
	dataPred.append(Output)
plt.plot(range(len(ts[:,3])),ts[:,3],'r', label='trained')
plt.plot(range(len(dataPred)),dataPred,'b', label='original')
plt.show()
plt.plot(range(len(epochMSE)),epochMSE)
plt.show()

# AdaptiveFIS.L2([7.1,3,5.9])
# AdaptiveFIS.L3()
# AdaptiveFIS.L4()
# OutputPred = AdaptiveFIS.Defuzzication()
[inputs2, sd2, mean2, consequencesPara2, y_out2, y_pred2,dE2]=AdaptiveFIS.ObtainParas(True)
print("consequences parameters: ", consequencesPara2)	





# # A simple fuzzy model describing how the heating power of a gas burner depends on the oxygen supply.
# FS = sf.FuzzySystem()

# # Define a linguistic variable.
# S_1 = sf.FuzzySet( points=[[0, 1.],  [1.2, 1.],  [1.5, 0]],          term="low_flow" )
# S_2 = sf.FuzzySet( points=[[0.5, 0], [1, 1], [2.5, 1], [3., 0]], term="medium_flow" )
# S_3 = sf.FuzzySet( points=[[5., 0],  [5.5, 1.], [6., 1.]],          term="high_flow" )
# FS.add_linguistic_variable("OXI", sf.LinguisticVariable( [S_1, S_2, S_3] ))

# # Define consequents.
# FS.set_crisp_output_value("LOW_POWER", 0)
# FS.set_crisp_output_value("MEDIUM_POWER", 1, True)
# FS.set_output_function("HIGH_FUN", "OXI*2")

# # Define fuzzy rules.
# RULE1 = "IF (OXI IS low_flow) THEN (POWER IS LOW_POWER)"
# RULE2 = "IF (OXI IS medium_flow) THEN (POWER IS MEDIUM_POWER)"
# RULE3 = "IF (OXI IS high_flow) THEN (POWER IS HIGH_FUN)"
# FS.add_rules([RULE1, RULE2, RULE3], False)

# # Set antecedents values, perform Sugeno inference and print output values.
# FS.set_variable("OXI", 1.44, True)
# print (FS.Sugeno_inference(['POWER'],False,False))