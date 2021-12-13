import os, re, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
sys.setrecursionlimit(int(1e6))
#Linear Fitting of x and y
def linearfitting(x,y):
    try:
      x1 = np.array(x)
      y1 = np.array(y)
    except:
      x1 = x
      y1 = y
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    sigmay = (sum((m*x1+c-y1)**2)/(len(x1)-2))**0.5
    dy = sum(y1**2)*len(y1)-sum(y1)**2
    dx = sum(x1**2)*len(x1)-sum(x1)**2
    R2 = (len(x1)*sum(x1*y1)-sum(x1)*sum(y1))**2/(dx*dy)
    sigma_a = (len(x1)*sigmay**2/dx)**0.5
    sigma_b = (sum(x1**2)*sigmay**2/dx)**0.5
    print('a={}    b={}    R2={}    sigma_a={}    sigma_b={}'.format(m, c, R2,sigma_a,sigma_b))
    return m,c,R2,sigma_a,sigma_b
#read output TXT
def readcv(path):
    params = {}
    with open(file=path) as f:
        for i in range(8):
            f.readline()
        for i in range(9):
            name, value = f.readline().split(' = ')
            try:
                params[name] = float(value)
            except:
                params[name] = value.strip('\n')
        while True:
            if 'Potential/V, Current/A, Charge/C' in f.readline():
                f.readline()
                break
        segment = {}
        segnum = 0
        while True:
            key = f.readline()
            if 'Segment' in key:
                segnum += 1
                seg = []
                
                while True:
                    ui = f.readline()
                    try:
                        seg.append(list(map(float,(ui.strip('\n')).split('\t'))))
                    except:
                        break
                segment[segnum] = np.array(seg)
            else:
                break
            #print(f.readline())
    return params, segment
#deprecated, do not use
def binarySearch(arr, l, r, x, err):
    if r >= l: 
        mid = int(l + (r - l)/2)
  
        if (arr[mid]-x)**2<=err**2: 
            return mid 

        elif (arr[mid]-x) > err: 
            return binarySearch(arr, l, mid-1, x, err) 
  
        else: 
            return binarySearch(arr, mid+1, r, x, err) 
  
    else: 
        return -1
#use this one, func should be increasing
def binarysearchfunc(func, low, high, x, err):
    mid = low +(high-low)/2
    #print(mid)
    if (func(mid)-x)**2<err**2:
        return mid
    elif (func(mid)-x)>err:
        return  binarysearchfunc(func, low, mid, x, err)
    else:
        return binarysearchfunc(func, mid, high, x, err)
if __name__=='__main__':
    err = 1e-10
    arr = np.arange(1,10,0.1)
    plt.style.use(['science','no-latex','nature'])
    #Load output files from experiments
    params1, segment1 = readcv('./1014/1014/0_1vs.txt')#0.1 V/s scan
    params2, segment2 = readcv('./1014/1014/steady.txt')#steady O2 
    params3, segment3 = readcv('./1014/1014/methanol_2.txt')#methanol
    params4, segment4 = readcv('./1014/1014/stir.txt')#stir and bubbling
    params5, segment5 = readcv('./1014/1014/bubb.txt')#only bubbling
    E = np.concatenate((segment1[9][:,0],segment1[10][:,0],segment1[9][0:1,0]))
    I_blank = -np.concatenate((segment1[9][:,1],segment1[10][:,1],segment1[9][0:1,1]))
    I_ORR = -np.concatenate((segment2[9][:,1],segment2[10][:,1],segment2[9][0:1,1]))
    I_ORR_stir = -np.concatenate((segment4[9][:,1],segment4[10][:,1],segment4[9][0:1,1]))
    I_ORR_bubb = -np.concatenate((segment5[9][:,1],segment5[10][:,1],segment5[9][0:1,1]))
    I_MOR = -np.concatenate((segment3[9][:,1],segment3[10][:,1],segment3[9][0:1,1]))
    fig5 = plt.figure()
    plt.grid(True)
    plt.plot(E, I_blank, label='blank')
    print(segment1[9][400,0])
    plt.hlines(y=-segment1[9][400,1],xmin = -0.29, xmax=0.2,linestyles='dashed',colors='green')
    plt.fill_between(segment1[9][:400,0],-segment1[9][:400,1],y2=-segment1[9][400,1],alpha = 0.5, color = 'violet')
    s = np.sum((segment1[9][:400,1]-segment1[9][400,1])*(segment1[9][1099,0]-segment1[9][1100,0]))
    plt.text(-0.25, -3.2e-6, 'desorption peak\n s = {:.3e} W'.format(s))
    plt.xlabel('E/V')
    plt.ylabel('I/A')
    plt.savefig('surface.png',dpi = 1000)
    fig =plt.figure()
    plt.grid(True)
    plt.plot(E, I_blank, label='blank')
    plt.plot(E, I_ORR,label='ORR')
    plt.xlabel('E/V')
    plt.ylabel('I/A')
    plt.legend()
    plt.savefig('ORR.png',dpi=1000)
    plt.cla()
    plt.grid(True)
    plt.plot(E, I_blank, label='blank')
    plt.plot(E, I_ORR_stir,label='stir')
    plt.plot(E, I_ORR_bubb,label='bubble')
    plt.plot(E, I_ORR,label='steady')
    plt.xlabel('E/V')
    plt.ylabel('I/A')
    plt.legend()
    plt.savefig('stir.png',dpi=1000)
    plt.cla()
    plt.grid(True)
    plt.plot(E, I_blank, label='blank')
    plt.plot(E, I_MOR,label='MOR')
    plt.xlabel('E/V')
    plt.ylabel('I/A')
    plt.legend()
    plt.savefig('MOR.png',dpi=1000)
    print(1)
    fig2 = plt.figure()
    I_red = segment2[10][:,1]-segment1[10][:,1]
    I_oxi = -segment3[9][:-1,1]+segment1[9][:-1,1]
    print(segment1[9][-1,0])
    CV_red = interp1d(segment1[10][:,0], I_red, kind='cubic')
    CV_oxi = interp1d(segment1[9][:-1,0],I_oxi, kind='cubic')
    E_oxi = binarysearchfunc(CV_oxi, 0, 0.4, 0, err)
    E_red =binarysearchfunc(lambda x:-CV_red(x), 0.4, 0.8, 0, err)
    E_max = binarysearchfunc(lambda x: CV_oxi(x)-CV_red(x), E_oxi, E_red, 0, err)
    I_max = CV_oxi(E_max)
    print('E_oxi:{}    E_red:{}    E_max:{}    I_max:{}'.format(E_oxi,E_red,E_max,I_max))
    plt.plot(segment1[10][:,0], I_red,label='ORR')
    plt.plot(segment1[9][:-1,0], I_oxi,label='MOR')
    #plt.plot(segment1[10][:,0],I_red-I_oxi[::-1])
    plt.hlines(0,-0.3,1.3,colors='black',linestyles='dashed')
    #plt.vlines(E_max,-2e-6,8e-6,linestyles='dashed')
    plt.xlim(-0.3, 1.3)
    plt.xlabel('E/V')
    plt.ylabel('I/A')
    plt.legend()
    plt.grid(True)
    plt.savefig('Irel.png',dpi=1000)
    plt.cla()
    plt.plot(segment1[10][600:850,0], np.log(np.abs(I_red[600:850])),label='ORR')
    print('start:{}     end:{}'.format(segment1[10][850,0],segment1[10][600,0]))
    a, b, R2, sigma_a, sigma_b = linearfitting(segment1[10][700:760,0],np.log(np.abs(I_red[700:760])))
    plt.plot(segment1[10][600:850,0],a*segment1[10][600:850,0]+b, label='linear fit')
    plt.text(0.46,-13,'y = {:.3e}x + {:.3e}'.format(a,b)+
    '\n'+r'$\eta$={:.2e} + {:.2e}'.format(b/a+E_red, -1/a)+r'$ \mathrm{lg} (I/\mathrm{A})$'
    +'\n'+r'$R^2={:.4f}$'.format(R2))
    print('start:{}     end:{}'.format(segment1[10][760,0],segment1[10][700,0]))
    plt.xlabel('E/V')
    plt.ylabel(r'$\mathrm{lg} (I/\mathrm{A})$')
    plt.legend()
    plt.grid(True)
    plt.savefig('E_lgI.png',dpi=1000)
    deltaE = []
    for i in np.linspace(err, I_max, 1000):
        E_r = binarysearchfunc(lambda x:-CV_red(x), E_max, E_red, -i, err)
        E_o = binarysearchfunc(CV_oxi, E_oxi, E_max, i, err)
        deltaE.append(E_r-E_o)
    fig3 = plt.figure()
    plt.plot(np.array(deltaE),np.linspace(0,I_max,1000)*np.array(deltaE))
    plt.xlabel('E/V')
    plt.ylabel('P/W')
    plt.ylim(0,2.5e-7)
    plt.savefig('P-E.png', dpi=1000)
    #plt.show()

    fig4 = plt.figure()
    array = np.array([1,1,2,1,2,1])
    plt.plot(np.arange(6),array)
    plt.xlim(0,6)
    plt.ylim(0,3)
    plt.xlabel('t')
    plt.ylabel('E')
    plt.tick_params(axis='both',labeltop = False, labelleft = False, labelright = False, labelbottom = False)
    plt.savefig('cv.png', dpi=1000)
