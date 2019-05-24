#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np
import matplotlib.pyplot as plt

from Running_Average_lib import Running_Average

#Pi_05=250

def noise(ma,sigma,size):
	return np.random.normal(ma,sigma,size)

def phase_def(Amin,Amax,Pi_05):
	indx=Amin if Amin<Amax else Amax
	return (indx+Pi_05) if (indx-Pi_05)<0 else (indx-Pi_05)
	#return ((indx-Pi_05)<0)?(indx+Pi_05):(indx-Pi_05)

def Urms(Dump):
	return np.sqrt(np.sum(np.square(Dump))/np.size(Dump))

def Sig(A,f,t,fi):
	return A*np.sin(2*np.pi*f*t+fi)

def Sig_trig(T,k,fi_rad):
	fi = int(round(fi_rad*T/(2*np.pi)))
	#print "fi ",fi
	p=float(T)/6.
	sig=np.zeros(2*T)
	#sigfi=np.zeros(T)
	c1=1
	c2=1
	for i in range(T):
		if i<=p or (i>=2*p and i<=4*p) or i>=6*p:
			sig[i]=0
		elif (i>p and i<1.5*p):
			
			sig[i]=k*c1
			c1+=1
		elif (i>=1.5*p and i<2.*p):
			sig[i]=k*c1	
			c1-=1
		elif (i>4*p and i<4.5*p):	
			sig[i]=-k*c2
			c2+=1
		elif (i>=4.5*p and i<5.*p):	
			sig[i]=-k*c2
			c2-=1			
		else:
			sig[i]=0

	sig[T:]=sig[:T]
	#sigfi=sig[fi:(T+fi)]		

	return sig[fi:(T+fi)]		 		



def Runn_Aver(MeanObj,Dump):
	RAM = []
	for d in Dump:
		MeanObj.add(d)
		RAM.append(MeanObj.get())
	return RAM	

def test_D_q(Alist,fi,f,sigma,size,dump_size,X):
	D_r_mean=[]
	D_a_mean=[]
	Q=[]
	#X=np.linspace(0,0.02,size)
	for A in Alist:
		D_r=[]
		D_a=[]
		for i in range(100):

			N=noise(0,sigma,size)
			

			if SIG_SIN:
				S=[Sig(A/2.,f,t,fi) for t in X]
			else:
				S=Sig_trig(size,A/2.,fi)	

			Y=S+N
			Dump=Y[::(size/dump_size)] #каждый 10й элемент
			
			Dumpa=Runn_Aver(MeanObj,Dump)
			ps=phase_def(np.argmin(S),np.argmax(S),size/4)
			pd=phase_def(np.argmin(Dump),np.argmax(Dump),dump_size/4)
			pa=phase_def(np.argmin(Dumpa),np.argmax(Dumpa),dump_size/4)
			
			dfi_s = ps*2./float(size)
			dfi_r = pd*2./float(dump_size)
			dfi_a = pa*2./float(dump_size)
			D_r.append(abs(dfi_s-dfi_r))
			D_a.append(abs(dfi_s-dfi_a))
			#D_r_mean.append(abs(dfi_s-dfi_r))
			#D_a_mean.append(abs(dfi_s-dfi_a))
		
		#Q.append(A)
		D_r_mean.append(np.mean(D_r))
		D_a_mean.append(np.mean(D_a))

	return D_r_mean,D_a_mean		




if __name__ =='__main__':
	fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5,figsize=(10, 10))
	ax1.set_title('Sig')
	ax1.set_xlabel('t')
	ax1.set_ylabel('U')
	ax2.set_title('Noise')
	ax2.set_xlabel('t')
	ax2.set_ylabel('U')
	ax3.set_title('Sig + Noise')
	ax3.set_xlabel('t')
	ax3.set_ylabel('U')
	ax4.set_title('Dump')
	ax4.set_xlabel('t')
	ax4.set_ylabel('U')
	ax5.set_title('Running Average')
	ax5.set_xlabel('t')
	ax5.set_ylabel('U')


	fig0, (ax0) = plt.subplots(nrows=1, ncols=1,figsize=(10, 10))

	ax0.set_title(u'Характеристика алгоритма')
	ax0.set_xlabel(u'сигнал/шум')
	ax0.set_ylabel(u'ошибка')
	
	figfi, (axfi) = plt.subplots(nrows=1, ncols=1,figsize=(10, 10))
	figfia, (axfia) = plt.subplots(nrows=1, ncols=1,figsize=(10, 10))

	axfi.set_title(u'Зависимость от начальной фазы без усреднения')
	axfi.set_xlabel(u'сигнал/шум')
	axfi.set_ylabel(u'ошибка')

	axfia.set_title(u'Зависимость от начальной фазы с усреднением')
	axfia.set_xlabel(u'сигнал/шум')
	axfia.set_ylabel(u'ошибка')

	Alist=[0.5,0.4,0.3,0.2,0.1,0.05,0.03,0.025,0.02,0.01,0.005,0.0025]
	MODEL_FI=True
	SIG_SIN = False

	A=0.01
	f=50
	sigma=0.0025
	size=1000
	dump_size=100
	MeanObj = Running_Average(10)

	fi=np.random.uniform(0,2*np.pi)
	N=noise(0,sigma,size)
	X=np.linspace(0,0.02,size)
	if SIG_SIN:
		S=[Sig(A/2.,f,t,fi) for t in np.linspace(0,0.02,size)]
	else:
		S=Sig_trig(size,0.00005,0)	
	Y=S+N
	Dump=Y[::(size/dump_size)] #каждый 10й элемент
	Xr=np.linspace(0,0.02,dump_size)
	Dumpa=Runn_Aver(MeanObj,Dump) #average


	print "Ideal min:",np.min(S)," argmin:",np.argmin(S)," max:",np.max(S)," argmax:",np.argmax(S)
	print "Dump min:",np.min(Dump)," argmin:",np.argmin(Dump)," max:",np.max(Dump)," argmax:",np.argmax(Dump)

	ps=phase_def(np.argmin(S),np.argmax(S),size/4)
	pd=phase_def(np.argmin(Dump),np.argmax(Dump),dump_size/4)
	pa=phase_def(np.argmin(Dumpa),np.argmax(Dumpa),dump_size/4)

	dfi_s = ps*2./float(size)
	dfi_r = pd*2./float(dump_size)
	dfi_a = pa*2./float(dump_size)

	print "phase ideal:",dfi_s," xi:",X[ps]," yi:",S[ps]," ymin:",np.min(S)," ymax:",np.max(S)," xmin",X[np.argmin(S)]," xmax:",X[np.argmax(S)]
	print "phase real:",dfi_r," xr:",Xr[pd]," yr:",Dump[pd]," ymin:",np.min(Dump)," ymax:",np.max(Dump)," xmin",Xr[np.argmin(Dump)]," xmax:",Xr[np.argmax(Dump)]
	print "phase average:",dfi_a," xr:",Xr[pa]," yr:",Dumpa[pa]," ymin:",np.min(Dumpa)," ymax:",np.max(Dumpa)," xmin",Xr[np.argmin(Dumpa)]," xmax:",Xr[np.argmax(Dumpa)]

	print "delta r:",abs(dfi_s-dfi_r)," a:",abs(dfi_s-dfi_a)
	#print "abs sum:",np.sum(np.abs(S)),np.sum(np.abs(S))/np.size(S)
	#print "square:",np.sqrt(np.sum(np.square(S))/np.size(S))
	print "Urms ideal",Urms(S)," Urms real:",Urms(Dump)

	ax1.plot(X, S, color='r', label='S')	
	ax2.plot(X, N, color='b', label='N')
	ax3.plot(X, Y, color='g', label='Y')
	ax4.plot(Xr, Dump, color='y', label='D')
	ax5.plot(Xr, Dumpa, color='y', label='RAM')

	print "fi0:",fi/np.pi

	D_r,D_a=test_D_q(Alist,fi,f,sigma,size,dump_size,X)

	#print len(D_r),len(D_a),len(Q)

	ax0.plot(Alist, D_r, color='r', label=u'Без усреднения')	
	ax0.plot(Alist, D_a, color='b', label=u'Скользящее усреднение')
	ax0.legend(loc='upper right')
	if MODEL_FI:
		Fi_r=[]
		Fi_a=[]
		clr=['#2f4f4f',\
				'#000000',\
				'#7cfc00',\
				'#87ceeb',\
				'#00ced1',\
				'#deb887',\
				'#6b8e23',\
				'#ffd700',\
				'#cd5c5c',\
				'#ff1493'
		]
		filist=np.linspace(0*np.pi,0.9*np.pi,10)
		for indx,fi in enumerate(filist):
			D_r1,D_a1=test_D_q(Alist,fi,f,sigma,size,dump_size,X)
			Fi_r.append(D_r1)
			Fi_a.append(D_a1)
			#print indx, clr[indx]
			axfi.plot(Alist, Fi_r[indx], c=clr[indx],lw=2,label=str(fi/np.pi)+u"п")
			axfia.plot(Alist, Fi_a[indx], c=clr[indx],lw=2,label=str(fi/np.pi)+u"п")
		axfi.legend(loc='upper right')	
		axfia.legend(loc='upper right')
	
	plt.show()