import numpy as np
import scipy.ndimage as sc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
from scipy import signal
import sys
import fabio

def getParabola(data, a = None, vx = None, vy = None):
	if (a != None and vx == None and vy == None):
		def polinom2(x, b, c):
			return a * x**2 + b * x + c
		coeffs = curve_fit(polinom2, data[:,0], data[:,1])[0]
		coeffs = np.array([a, coeffs[0], coeffs[1]])
	elif (a == None and vx != None and vy == None):
		def polinom2(x, a, c):
			return a * x**2 + (-vx * 2 * a) * x + c
		coeffs = curve_fit(polinom2, data[:,0], data[:,1])[0]
		coeffs = np.array([coeffs[0], (-vx * 2 * coeffs[0]), coeffs[1]])
	elif (a != None and vx != None and vy == None):
		def polinom2(x, c):
			return a * x**2 + (-vx * 2 * a) * x + c
		coeffs = curve_fit(polinom2, data[:,0], data[:,1])[0]
		coeffs = np.array([a, (-vx * 2 * a), coeffs[0]])
	elif (a != None and vx != None and vy != None):
		coeffs = np.array([a, (-vx * 2 * a), (vy + (-vx * 2 * a)**2 / (4 * a))])
	else:
		def polinom2(x, a, b, c):
			return a * x**2 + b * x + c
		coeffs = curve_fit(polinom2, data[:,0], data[:,1])[0]
		coeffs = np.array([coeffs[0], coeffs[1], coeffs[2]])
	
	f = np.poly1d(coeffs)
	y = f(data[:,0])
	yc = np.sum(data[:,1]) / np.size(data[:,1])
	RSS = np.sum((data[:,1] - y)**2)
	TSS = np.sum((data[:,1] - yc)**2)
	R2 = 1 - RSS / TSS
	return coeffs, RSS, R2

def rotatePoints(data, angle):
	angle = np.pi*angle/180
	mapMatrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
	src = data#.T
	dst = np.dot(mapMatrix, src)
	return dst.T

def getAngle(x, y, startAngle):
	data = np.array([x,y])
	arr = []
	angles = np.linspace(-startAngle, startAngle, 2 * startAngle + 1)
	for x in angles:
		y = getParabola(rotatePoints(data, x))[2]
		arr.append([x, y])
	R2 = np.array(arr)
	minIndR2 = np.where(R2[:,1] == np.max(R2[:,1]))[0][0]
	startAngle = R2[minIndR2][0]

	l = getParabola(rotatePoints(data, startAngle + 1))[2]
	la = startAngle + 1
	c = getParabola(rotatePoints(data, startAngle))[2]
	ca = startAngle
	r = getParabola(rotatePoints(data, startAngle - 1))[2]
	ra = startAngle - 1
	while (1):
		if (l < r):
			la = ca
			l = c
			ca = ca + (ra - ca) / 2.
			c = getParabola(rotatePoints(data, ca))[2]
		elif (l > r):
			ra = ca
			r = c
			ca = ca + (la - ca) / 2.
			c = getParabola(rotatePoints(data, ca))[2]
		else:
			break
	return ca

def error(x, y, y3, R, x0, y0):
	x = x.T
	a = 1/(2*R)
	b = x0/R
	c = x0**2/2/R+y0
	F = np.array([(x-x)+1, x, x**2]).T #Матрица плана
	Ft = F.T #Транспонированная матрица плана
	G = np.dot(Ft,F) #информационная матрица
	C = np.linalg.inv(G) #Матрица дисперсии
	stda = (y-y3).std()
	da=((2.85*stda*np.sqrt(C[2,2]))*100/a)
	dc=((2.85*stda*np.sqrt(C[0,0]))*100/c)
	db=((2.85*stda*np.sqrt(C[1,1]))*100/b)
	dR = np.abs(da/2/a**2)
	dx0 = np.abs(R*db)+np.abs(x0/R*dR)
	dy0 = np.abs(c)+np.abs(x0/R*dx0)+np.abs(x0**2/2/R**2*dR)
	return da, db, dc


def parabola(x, R, x0, y0):
	#R = -50
	return 1/(2*R)*(x-x0)**2 + y0

def press(event):
	global x,y,xn,yn,u,n,popt,flag
	sys.stdout.flush()
	ax_axis = ax.axis()
	#ax_axis2 = ax2.axis()
	ax.cla()
	ax2.cla()
	ax3.cla()
	ax.imshow(img, cmap='Greys_r', interpolation='nearest', extent = extent, vmin = vmin, vmax = vmax) # vmin=600, vmax=3000
	
	
	if event.key == 'right':
		n += 1
		n = n%u
		nn = np.abs(n)
		print(nn)
		ax_axis = np.array([-np.abs(ax_axis[1] - ax_axis[0])/2, np.abs(ax_axis[1] - ax_axis[0])/2, -np.abs(ax_axis[3] - ax_axis[2])/2, np.abs(ax_axis[3] - ax_axis[2])/2]) + np.array([x[nn],x[nn],y[nn],y[nn]])
	elif event.key == 'left':
		n -= 1
		n = n%u
		nn = np.abs(n)
		print(nn)
		ax_axis = np.array([-np.abs(ax_axis[1] - ax_axis[0])/2, np.abs(ax_axis[1] - ax_axis[0])/2, -np.abs(ax_axis[3] - ax_axis[2])/2, np.abs(ax_axis[3] - ax_axis[2])/2]) + np.array([x[nn],x[nn],y[n],y[nn]])
	elif event.key == 'enter':
		u = 50
	elif event.key == 'ctrl+enter':
		flag = True
	elif event.key == 'shift+enter':
		x = np.append(x, event.xdata)
		y = np.append(y, event.ydata)
	
	if np.size(x) >= 3:
		if np.size(x) == u+1:
			len = (x[:-1]-x[u])**2 + (y[:-1]-y[u])**2
			x = np.delete(x, np.where(len == np.min(len))[0][0])
			y = np.delete(y, np.where(len == np.min(len))[0][0])
			p = x.argsort()
			n = np.where(p == u-1)[0][0]
			print(n)
			x = x[p]
			y = y[p]
			
		
		angle = getAngle(x, y, 15)
		print(angle)
		result = rotatePoints(np.array([x,y]), angle)
		x2 = result[:,0]
		y2 = result[:,1]
		popt, pcov = curve_fit(parabola, x2, y2, p0=popt)
		perr = np.sqrt(np.diag(pcov))
		p = x2.argsort()
		x2 = x2[p]
		y2 = y2[p]
		#popt = [-50, 9.743047575864617, 64.85125911745224]
		x3 = np.linspace(np.min(x2)-(np.min(x2)+np.max(x2)), np.max(x2)+(np.min(x2)+np.max(x2)), 1000)
		#x3 = np.linspace(2*np.min(x2), 2*np.max(x2), 100)
		y3 = parabola(x3, *popt)
		print(*popt)
		result = rotatePoints(np.array([x3,y3]), -angle)
		x3 = result[:,0]
		y3 = result[:,1]
		if flag:
			ax.plot(x3,y3,"-")
		
		norm = ((x2[n]-popt[1])**2/popt[0]**2+1**2)**0.5
		(xn, yn) = (-(x2[n]-popt[1])/popt[0], 1)/norm
		xn = np.linspace(-xn, xn, int(np.abs(int(popt[0]/pixel_size*1E-6))/2)*20+1)*popt[0]*0.5 + x2[n]
		yn = np.linspace(-yn, yn, int(np.abs(int(popt[0]/pixel_size*1E-6))/2)*20+1)*popt[0]*0.5 + y2[n]
		result_n = rotatePoints(np.array([xn,yn]), -angle)
		xn = result_n[:,0]
		yn = result_n[:,1]
		ax.plot(xn,yn,"w-o")
		I = img[ny - 1 - np.asarray(((yn - extent[2])/pixel_size*1E-6), dtype = np.int),np.asarray(((xn - extent[0])/pixel_size*1E-6), dtype = np.int) ]
		In = img[ny - 1 - np.asarray(((y[n] - extent[2])/pixel_size*1E-6), dtype = np.int),np.asarray(((x[n] - extent[0])/pixel_size*1E-6), dtype = np.int) ]
		#I = img[np.asarray(np.abs((xn - np.min(x))/pixel_size*1E-6), dtype = np.int),np.asarray(np.abs((yn - np.min(y))/pixel_size*1E-6), dtype = np.int)]
		nlen = ((np.min(xn)-np.max(xn))**2+(np.min(yn)-np.max(yn))**2)**0.5
		ax3.plot(np.linspace(-nlen/2,nlen/2,np.size(xn)),I,"-")
		nlen2 = ((np.min(xn)-x[n])**2+(np.min(yn)-y[n])**2)**0.5
		ax3.plot(-nlen/2+nlen2,In,"ro")
		
		#print(x, np.array([x2,parabola(x2, *popt)])[1,:])
		dR, dx0, dy0 = error(x2,parabola(x2, *popt),y2, *popt)
		#int(np.abs((xn - np.min(x))/pixel_size*1E-6))
		#int(np.abs((yn - np.min(y))/pixel_size*1E-6))
		
		popt_print = rotatePoints(np.array([[popt[1]],[popt[2]]]), -angle)
		if (np.size(x) == 4) and (u != 4):
			x = np.linspace(np.min(x2), np.max(x2), u)
			y = parabola(x, *popt)
			result = rotatePoints(np.array([x,y]), -angle)
			popt_print = rotatePoints(np.array([[popt[1]],[popt[2]]]), -angle)
			x = result[:,0]
			y = result[:,1]
		print("gkjgkjgk ", popt_print)
		ax.text(ax_axis[0]+(ax_axis[1]-ax_axis[0])*0.1, ax_axis[3]-(ax_axis[3]-ax_axis[2])*1.1, "$\\alpha$: %0.3f\nR: %0.3f +- %0.3f $\\mu m$\n v: %0.3f +- %0.3f $\\mu m$, %0.3f +- %0.3f $\\mu m$"%(angle, np.abs(popt[0]),perr[0], popt_print[0,0], perr[1], popt_print[0,1], perr[2]))
		
		ax2.plot(x2,parabola(x2, *popt)-y2,"-.")
		
	ax.plot(x,y,"ro")
	ax.axis(ax_axis)
	#ax2.axis(ax_axis2)
	fig.canvas.draw()

x = np.array([])
y = np.array([])
xn = np.array([])
yn = np.array([])

#max и min значения серого
vmin = 4908
vmax = 7138

popt = np.array([-50, 16, 56])	 # "-" в первом индексе для ветвей вниз #253.448
file = 'C:/Users/askor/OneDrive/Рабочий стол/Xoptics/2024/CRL Radiography/09.12 (G-series)/G006_0deg_20241209-120205_MicroLine ML8051 ML1553714 256 _3296x2472_t60_a1_g1-raw.tif'
pixel_size = 2.0068E-6
img = fabio.open(file)
img = np.asarray(img.data)
#img[img>4000]=1000
img = img.T

nx = np.size(img[0,:])
ny = np.size(img[:,0])

#fig, (ax, ax2, ax3) = plt.subplots(3,1,2)
fig = plt.figure()
ax = plt.subplot(1,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,2,4)
fig.canvas.mpl_connect('key_press_event', press)
#fig.canvas.mpl_connect('button_press_event', press)
#extent = np.array([-(nx+1)/2, (nx-1)/2, -(ny+1)/2, (ny-1)/2])*pixel_size/1E-6
extent = np.array([-(nx)/2, (nx)/2, -(ny)/2, (ny)/2])*pixel_size/1E-6
u = 50 #кол-во точек параболы
n =0
flag = False
#print(extent)
#extent = np.array([0, ny, nx, 0])-0.5
ax.imshow(img, cmap='Greys_r', interpolation='nearest', extent=extent, vmin = vmin, vmax = vmax) # vmin=600, vmax=3000    vmin = 36079, vmax = 50841
plt.show()
