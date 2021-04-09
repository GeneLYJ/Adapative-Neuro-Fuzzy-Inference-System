# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 18:08:32 2020

@author: Lenovo
"""
from numpy import array, linspace
import numpy as np
import math

# L1: Input Layer
# L2: Fuzzication Layer
# L3: Rule Layer
# L4: Normalized Layer
# L5: Defuzzication Layer

class Anfis(object):
    
	def __init__(self, n_inputs=None, sd=None, mean=None, rules=None, parameter=None\
			  ,consequences=None, yOutput=None, learningRate = None,\
				  memFuncType=None, z=None):
        
		self._n = np.reshape(np.repeat(n_inputs,3),(3,3))
		

		if sd is not None and mean is not None:
			self._x = sd
			self._z = mean
			self._rules = rules
			self._paras = parameter
			self._conseq = consequences
			self._yO = yOutput
			self._LR =learningRate
			self._mfType = memFuncType
			self._zParam = z

			
			
	def TrainingSet(self,n_inputs=None,sd=None,mean=None,z=None, yOutput=None):
		
		self._n = np.reshape(np.repeat(n_inputs,3),(3,3))
		self._x = sd
		self._z = mean
		self._zParam = z
		self._yO = yOutput
			
		
	def L2(self,n_inputs=None):
		self.membership = np.zeros((3,3))
		self.dE_dx = np.zeros((2,3,3))
		
		if n_inputs is not None:
			self._n = np.reshape(np.repeat(n_inputs,3),(3,3))
		
		for row in range(len(self._x)):
			for col in range(len(self._x[0])):
				#print(self._mfType[col],self._n[row][col],self._z[row][col],self._x[row][col])
				
				m = self._z[row][col]
				s = self._x[row][col]
				x = self._n[row][col]
				
				if self._mfType[col] == 1:
					alpha = math.exp(-0.5*(((x-m))/s)**2)
					
					dE1 = (x - m)/(s**2)
						
					dE2 = ((x - m)**2)/(s**3)
						
				if self._mfType[col] == 2:
					alpha = 1/(1 + math.exp(s*(x - m)))
						
					dE1 = math.exp(s*(x - m))*s*alpha
						
					dE2 = math.exp(s*(x - m))*(m-x)*alpha
				
				if self._mfType[col] == 3:
					alpha = 1/(1 + math.exp(s*(m - x)))
						
					dE1 = -math.exp(s*(m - x))*s*alpha
						
					dE2 = math.exp(s*(m - x))*(x-m)*alpha

				self.membership[row][col] = alpha
				self.dE_dx[0,row,col] = dE1
				self.dE_dx[1,row,col] = dE2

		return self.membership

	def L3(self, mf=None):
		self.tnorm = []
		
		if mf is None:
			self.mf = self.membership
		else:
			self.mf = mf
		
		length = len(self.mf)
		
		self.tnorm.append(np.tile(self.mf[0], 3**(length-1)))
		self.tnorm.append(np.repeat(self.mf[1], 3**(length-1)))
		temp = np.repeat(self.mf[2], 3**(length-2))
		self.tnorm.append(np.tile(temp, 3**(length-2)))
		first = np.multiply(self.tnorm[0], self.tnorm[1])
		self.y_ith = []
		self.y_ith = np.multiply(first, self.tnorm[2])

	def L4(self, text=False):
		# Normalized
		sum_Weight = np.sum(self.y_ith)
		normalized = self.y_ith/sum_Weight
		self.normalized = normalized
		
		if text is True:
			row_format = "{:<8} {:<8} {:<4} {:<4} {:<4}"
			print(row_format.format("IF1","IF2","IF3","Output","Normalized"))
			row_format = "{:.4f}, {:.4f}, {:.4f} {:.4f} {:.4f}"
			
			for x in range(len(self.tnorm[1])):
				#print(row_format.format(self.tnorm[0][x]))
				print(row_format.format(self.tnorm[0][x], self.tnorm[1][x], self.tnorm[2][x],\
							self.y_ith[x],self.normalized[x]))
		#return normalized
		
	def Defuzzication(self):
		
		self.y_Pred = 0
		for i in range(len(self.then)):
			y_Pred = self.normalized[i]*float(self._zParam[self.then[i]-1])
			self.y_Pred += y_Pred
			#print(self._zParam[self.then[i]-1])
		return self.y_Pred
		
	def MSE(self, text=False, yOutput=None):
		# measure of error:
		#moe = 0.5*(self._yO-self.y_Pred)**2
		self.zPrev = []
		if yOutput is not None:
			self._yO = yOutput
		
		LR_er = self._LR*(self.y_Pred-self._yO)		
		for i in range(len(self.then)):
			self.zPrev.append(self._zParam[self.then[i]-1])
			self._zParam[self.then[i]-1]=self._zParam[self.then[i]-1]\
				- (LR_er*(self.normalized[i]))
		
		if text is True:
			print("consequences parameters: ",self._zParam)		
		
		return (self._yO-self.y_Pred)**2
	
	def BackPass(self):
		length = len(self._n)		
		sigma = self._LR*self.y_Pred-self._yO
		dO_dsORdm = np.multiply(self.zPrev,self.normalized)		

		for col in range(len(self._n[0])):
			st = np.zeros(3)
			
			
			st[col] = self.dE_dx[0,0,col]
			row1_m = np.tile(st, 3**(length-1))
			dE = np.dot(row1_m, dO_dsORdm)
			self._z[0][col] = self._z[0][col] - sigma*dE

			st[col] = self.dE_dx[1,0,col]
			row1_sd = np.tile(st, 3**(length-1))
			dE = np.dot(row1_sd, dO_dsORdm)
			self._x[0][col] = self._x[0][col] - sigma*dE
			
			st[col] = self.dE_dx[0,1,col]
			row2_m = np.repeat(st, 3**(length-1))
			dE = np.dot(row2_m, dO_dsORdm)
			self._z[1][col] = self._z[1][col] - sigma*dE
			
			st[col] = self.dE_dx[1,1,col]
			row2_sd = np.repeat(st, 3**(length-1))
			dE = np.dot(row2_sd, dO_dsORdm)
			self._x[1][col] = self._x[1][col] - sigma*dE
			
			st[col] = self.dE_dx[0,2,col]
			row3_m = np.tile(np.repeat(st, 3**(length-2)), 3**(length-2))
			dE = np.dot(row3_m, dO_dsORdm)
			self._z[2][col] = self._z[2][col] - sigma*dE
			
			st[col] = self.dE_dx[1,2,col]
			row3_sd = np.tile(np.repeat(st, 3**(length-2)), 3**(length-2))
			dE = np.dot(row3_sd, dO_dsORdm)
			self._x[2][col] = self._x[2][col] - sigma*dE
		
		return self._z, self._x
		
	def If_Then(self):
		rulesTxt = []
			
		length = len(self._rules)
			
		rulesTxt.append(np.tile(self._rules[0],3**(length-1)))
		rulesTxt.append(np.repeat(self._rules[1], 3**(length -1)))
		temp = np.repeat(self._rules[2], 3**(length-2))
		rulesTxt.append(np.tile(temp, 3**(length - 2)))
		
		# consequences for 27 rules placement. Edit here:
		self.then = [1,2,1,1,1,1,2,3,3,1,2,3,2,3,4,3,4,5,3,3,4,4,5,6,4,5,6]
			
		# print string
		
		row_format = "{:<12} {:<12} {:<12} {:<12}"
		
		print(row_format.format("IF", "IF", "IF", "THEN"))
		print(row_format.format(self._paras[0], self._paras[1], self._paras[2],\
						  self._paras[3]))
		for x in range(len(self.then)):
			 print(row_format.format(rulesTxt[0][x], rulesTxt[1][x],\
						 rulesTxt[2][x], self._conseq[self.then[x]-1]))
		
	def ObtainParas(self,paras=False):
		if paras is True:
			return self._n,self._x,self._z,self._zParam,self._yO,self.y_Pred,self.dE_dx
	
		
		