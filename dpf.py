import cv2
import numpy, math

import parser
import argparse
from argparse import ArgumentParser, ArgumentTypeError
import re, torch, torchvision, sys, validate
from tqdm import tqdm


def parseNumList(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    # ^ (or use .split('-'). anyway you like.)
    if not m:
        raise ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start,10), int(end,10)+1))


class tamLados(): 

    def __init__(self, heighU=0, heighD=0, widthL=0, widthR=0):
        self.altC = heighU
        self.altB = heighD
        self.largE = widthL
        self.largD = widthR 

    def __str__(self):
        message = f"tamLados: [ altC:{self.altC}, altB: {self.altB}, largE: {self.largE}, largD: {self.largD}]"
        return message
        
        
def binarizacaoOtsuGlobal(img, imgVar, chave): 

    w0 = None; w1 = None; want = None; u0 = None; u1 = None; u2 = None; ut = None; 
    uant = None; sigt = None; melhorvariancia = None; varianciaOtsu = None;
    melhorlimiar, i, j, t, = None , None, None, None
    histograma = []
    w, h = img.shape[1], img.shape[0]
    t = w*h  
    ut, sigt = 0, 0 

    for i in range(256): 
        histograma.append(0)

    t = 0 
    for i in range(h): 
        for j in range(w):
            if chave == 1: 

                if imgVar[i][j] == 255: 
                    histograma[img[i][j]] += 1 
                    t += 1 

            else: 
                
                histograma[img[i][j]] += 1 
                t += 1 

    for i in range(256): 
        ut += numpy.float32(i) * numpy.float32(histograma[i]) / numpy.float32(t) 

    for i in range(256): 
        sigt += numpy.float32(i-ut) * (numpy.float32(i-ut)*numpy.float32(histograma[i])/numpy.float32(t))

    melhorvariancia = -999.99
    want = 0.0; uant = 0.0 

    for i in range(256): 
        u2=0.0; u0=0.0; u1=0.0; w0=0.0; w1=0.0;
        w0=want+numpy.float32(histograma[i])/numpy.float32(t);
        u2=uant+numpy.float32(i)*numpy.float32(histograma[i])/numpy.float32(t);
        want=w0;
        uant=u2;
        w1=1-w0;

        if w0!=0:
            u0 = u2/w0;
        if w1!=0:
            u1 = (ut-u2)/w1;

        varianciaOtsu=w0*w1*(u0-u1)*(u0-u1)/sigt;

        if(varianciaOtsu>melhorvariancia):
            melhorvariancia=varianciaOtsu;
            melhorlimiar=i;
        
    return melhorlimiar;

RY = [[-1,-1,0,1,1,1,0,-1],[-2,-1,0,1,2,1,0,-1],[-2,-2,-1,0,1,2,2,2,1,0,-1,-2],[-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3],[-4,-4,-4,-3,-3,-2,-2,-1,0,1,2,2,3,3,4,4,4,4,4,3,3,2,2,1,0,-1,-2,-2,-3,-3,-4,-4],[-5,-5,-5,-4,-3,-2,-1,0,1,2,3,4,5,5,5,5,5,4,3,2,1,0,-1,-2,-3,-4,-5,-5],[-6,-6,-6,-5,-5,-4,-4,-3,-2,-1,0,1,2,3,4,4,5,5,6,6,6,6,6,5,5,4,4,3,2,1,0,-1,-2,-3,-4,-4,-5,-5,-6,-6],[-7,-7,-7,-6,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,6,7,7,7,7,7,6,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-6,-7,-7],[-8,-8,-8,-7,-7,-6,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,6,7,7,8,8,8,8,8,7,7,6,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-6,-7,-7,-8,-8],[-9,-9,-9,-9,-8,-8,-8,-7,-7,-6,-5,-5,-4,-3,-3,-2,-1,0,1,2,3,3,4,5,5,6,7,7,8,8,8,9,9,9,9,9,9,9,8,8,8,7,7,6,5,5,4,3,3,2,1,0,-1,-2,-3,-3,-4,-5,-5,-6,-7,-7,-8,-8,-8,-9,-9,-9],[-10,-10,-10,-10,-9,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,9,10,10,10,10,10,10,10,9,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-9,-10,-10,-10],[-11,-11,-11,-11,-10,-10,-9,-9,-8,-8,-7,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,7,8,8,9,9,10,10,11,11,11,11,11,11,11,10,10,9,9,8,8,7,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-7,-8,-8,-9,-9,-10,-10,-11,-11,-11],[-12,-12,-12,-12,-11,-11,-10,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,10,11,11,12,12,12,12,12,12,12,11,11,10,10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-10,-11,-11,-12,-12,-12],[-13,-13,-13,-13,-12,-12,-12,-11,-11,-10,-10,-9,-9,-8,-7,-6,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,6,7,8,9,9,10,10,11,11,12,12,12,13,13,13,13,13,13,13,12,12,12,11,11,10,10,9,9,8,7,6,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-6,-7,-8,-9,-9,-10,-10,-11,-11,-12,-12,-12,-13,-13,-13],[-14,-14,-14,-14,-13,-13,-13,-12,-12,-11,-11,-10,-9,-8,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,8,9,10,11,11,12,12,13,13,13,14,14,14,14,14,14,14,13,13,13,12,12,11,11,10,9,8,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-8,-9,-10,-11,-11,-12,-12,-13,-13,-13,-14,-14,-14],[-15,-15,-15,-15,-14,-14,-14,-13,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,13,14,14,14,15,15,15,15,15,15,15,14,14,14,13,13,12,11,10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-13,-14,-14,-14,-15,-15,-15],[-16,-16,-16,-16,-16,-15,-15,-15,-14,-14,-13,-13,-12,-12,-11,-11,-10,-10,-9,-8,-7,-6,-5,-4,-4,-3,-2,-1,0,1,2,3,4,4,5,6,7,8,9,10,10,11,11,12,12,13,13,14,14,15,15,15,16,16,16,16,16,16,16,16,16,15,15,15,14,14,13,13,12,12,11,11,10,10,9,8,7,6,5,4,4,3,2,1,0,-1,-2,-3,-4,-4,-5,-6,-7,-8,-9,-10,-10,-11,-11,-12,-12,-13,-13,-14,-14,-15,-15,-15,-16,-16,-16,-16],[-17,-17,-17,-17,-17,-16,-16,-16,-15,-15,-15,-14,-14,-13,-12,-11,-10,-9,-9,-8,-7,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,7,8,9,9,10,11,12,13,14,14,15,15,15,16,16,16,17,17,17,17,17,17,17,17,17,16,16,16,15,15,15,14,14,13,12,11,10,9,9,8,7,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-7,-8,-9,-9,-10,-11,-12,-13,-14,-14,-15,-15,-15,-16,-16,-16,-17,-17,-17,-17],[-18,-18,-18,-18,-18,-17,-17,-17,-16,-16,-15,-14,-14,-13,-13,-12,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,12,13,13,14,14,15,16,16,17,17,17,18,18,18,18,18,18,18,18,18,17,17,17,16,16,15,14,14,13,13,12,12,11,10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-12,-13,-13,-14,-14,-15,-16,-16,-17,-17,-17,-18,-18,-18,-18],[-19,-19,-19,-19,-19,-18,-18,-18,-17,-17,-16,-16,-15,-15,-14,-13,-12,-11,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,11,12,13,14,15,15,16,16,17,17,18,18,18,19,19,19,19,19,19,19,19,19,18,18,18,17,17,16,16,15,15,14,13,12,11,11,10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-11,-12,-13,-14,-15,-15,-16,-16,-17,-17,-18,-18,-18,-19,-19,-19,-19],[-20,-20,-20,-20,-20,-19,-19,-19,-18,-18,-17,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,17,18,18,19,19,19,20,20,20,20,20,20,20,20,20,19,19,19,18,18,17,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-17,-18,-18,-19,-19,-19,-20,-20,-20,-20]];
RX = [[0,1,1,1,0,-1,-1,-1 ],[0,1,2,1,0,-1,-2,-1],[0,1,2,2,2,1,0,-1,-2,-2,-2,-1 ],[0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1 ],[0,1,2,2,3,3,4,4,4,4,4,3,3,2,2,1,0,-1,-2,-2,-3,-3,-4,-4,-4,-4,-4,-3,-3,-2,-2,-1 ],[0,1,2,3,4,5,5,5,5,5,4,3,2,1,0,-1,-2,-3,-4,-5,-5,-5,-5,-5,-4,-3,-2,-1 ],[0,1,2,3,4,4,5,5,6,6,6,6,6,5,5,4,4,3,2,1,0,-1,-2,-3,-4,-4,-5,-5,-6,-6,-6,-6,-6,-5,-5,-4,-4,-3,-2,-1 ],[0,1,2,3,4,5,6,6,7,7,7,7,7,6,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-6,-7,-7,-7,-7,-7,-6,-6,-5,-4,-3,-2,-1 ],[0,1,2,3,4,5,6,6,7,7,8,8,8,8,8,7,7,6,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-6,-7,-7,-8,-8,-8,-8,-8,-7,-7,-6,-6,-5,-4,-3,-2,-1 ],[0,1,2,3,3,4,5,5,6,7,7,8,8,8,9,9,9,9,9,9,9,8,8,8,7,7,6,5,5,4,3,3,2,1,0,-1,-2,-3,-3,-4,-5,-5,-6,-7,-7,-8,-8,-8,-9,-9,-9,-9,-9,-9,-9,-8,-8,-8,-7,-7,-6,-5,-5,-4,-3,-3,-2,-1 ],[0,1,2,3,4,5,6,7,8,9,9,10,10,10,10,10,10,10,9,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-9,-10,-10,-10,-10,-10,-10,-10,-9,-9,-8,-7,-6,-5,-4,-3,-2,-1 ],[0,1,2,3,4,5,6,7,7,8,8,9,9,10,10,11,11,11,11,11,11,11,10,10,9,9,8,8,7,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-7,-8,-8,-9,-9,-10,-10,-11,-11,-11,-11,-11,-11,-11,-10,-10,-9,-9,-8,-8,-7,-7,-6,-5,-4,-3,-2,-1 ],[0,1,2,3,4,5,6,7,8,9,10,10,11,11,12,12,12,12,12,12,12,11,11,10,10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-10,-11,-11,-12,-12,-12,-12,-12,-12,-12,-11,-11,-10,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1 ],[0,1,2,3,4,5,6,6,7,8,9,9,10,10,11,11,12,12,12,13,13,13,13,13,13,13,12,12,12,11,11,10,10,9,9,8,7,6,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-6,-7,-8,-9,-9,-10,-10,-11,-11,-12,-12,-12,-13,-13,-13,-13,-13,-13,-13,-12,-12,-12,-11,-11,-10,-10,-9,-9,-8,-7,-6,-6,-5,-4,-3,-2,-1 ],[0,1,2,3,4,5,6,7,8,8,9,10,11,11,12,12,13,13,13,14,14,14,14,14,14,14,13,13,13,12,12,11,11,10,9,8,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-8,-9,-10,-11,-11,-12,-12,-13,-13,-13,-14,-14,-14,-14,-14,-14,-14,-13,-13,-13,-12,-12,-11,-11,-10,-9,-8,-8,-7,-6,-5,-4,-3,-2,-1 ],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,13,14,14,14,15,15,15,15,15,15,15,14,14,14,13,13,12,11,10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-13,-14,-14,-14,-15,-15,-15,-15,-15,-15,-15,-14,-14,-14,-13,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1 ],[0,1,2,3,4,4,5,6,7,8,9,10,10,11,11,12,12,13,13,14,14,15,15,15,16,16,16,16,16,16,16,16,16,15,15,15,14,14,13,13,12,12,11,11,10,10,9,8,7,6,5,4,4,3,2,1,0,-1,-2,-3,-4,-4,-5,-6,-7,-8,-9,-10,-10,-11,-11,-12,-12,-13,-13,-14,-14,-15,-15,-15,-16,-16,-16,-16,-16,-16,-16,-16,-16,-15,-15,-15,-14,-14,-13,-13,-12,-12,-11,-11,-10,-10,-9,-8,-7,-6,-5,-4,-4,-3,-2,-1 ],[0,1,2,3,4,5,6,7,7,8,9,9,10,11,12,13,14,14,15,15,15,16,16,16,17,17,17,17,17,17,17,17,17,16,16,16,15,15,15,14,14,13,12,11,10,9,9,8,7,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-7,-8,-9,-9,-10,-11,-12,-13,-14,-14,-15,-15,-15,-16,-16,-16,-17,-17,-17,-17,-17,-17,-17,-17,-17,-16,-16,-16,-15,-15,-15,-14,-14,-13,-12,-11,-10,-9,-9,-8,-7,-7,-6,-5,-4,-3,-2,-1],[0,1,2,3,4,5,6,7,8,9,10,11,12,12,13,13,14,14,15,16,16,17,17,17,18,18,18,18,18,18,18,18,18,17,17,17,16,16,15,14,14,13,13,12,12,11,10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-12,-13,-13,-14,-14,-15,-16,-16,-17,-17,-17,-18,-18,-18,-18,-18,-18,-18,-18,-18,-17,-17,-17,-16,-16,-15,-14,-14,-13,-13,-12,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1 ],[0,1,2,3,4,5,6,7,8,9,10,11,11,12,13,14,15,15,16,16,17,17,18,18,18,19,19,19,19,19,19,19,19,19,18,18,18,17,17,16,16,15,15,14,13,12,11,11,10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-11,-12,-13,-14,-15,-15,-16,-16,-17,-17,-18,-18,-18,-19,-19,-19,-19,-19,-19,-19,-19,-19,-18,-18,-18,-17,-17,-16,-16,-15,-15,-14,-13,-12,-11,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1 ],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,17,18,18,19,19,19,20,20,20,20,20,20,20,20,20,19,19,19,18,18,17,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-17,-18,-18,-19,-19,-19,-20,-20,-20,-20,-20,-20,-20,-20,-20,-19,-19,-19,-18,-18,-17,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]];

tamRaio = [8,8,12,16,32,28,40,40,48,68,56,72,68,88,88,84,112,112,112,116,112];


def centroGeomPoro(imgPoros, y, x, posY, posX):
    cont = 1;
    if (y >=1 and y < imgPoros.shape[0]-1 and x >=1 and x < imgPoros.shape[1]-1):
        imgPoros[y][x] = 0
        posY[0] += y 
        posX[0] += x 

        if (imgPoros[y-1][x-1] == 255):
            cont += centroGeomPoro(imgPoros,y-1,x-1, posY, posX);
        if imgPoros[y][x-1] == 255:
            cont += centroGeomPoro(imgPoros,y,x-1, posY, posX);
        if imgPoros[y+1][x-1] == 255:
            cont += centroGeomPoro(imgPoros,y+1,x-1, posY, posX);
        if imgPoros[y-1][x] == 255:
            cont += centroGeomPoro(imgPoros,y-1,x, posY, posX);
        if imgPoros[y+1][x] == 255:
            cont += centroGeomPoro(imgPoros,y+1,x, posY, posX);
        if imgPoros[y-1][x+1] == 255:
            cont += centroGeomPoro(imgPoros,y-1,x+1, posY, posX);
        if imgPoros[y][x+1] == 255:
            cont += centroGeomPoro(imgPoros,y,x+1, posY, posX);
        if imgPoros[y+1][x+1] == 255:
            cont += centroGeomPoro(imgPoros,y+1,x+1, posY, posX);

        return cont 

    return 0

class POROS():

    def __init__(self, y=0, x=0):
        self.y = y 
        self.x = x


def poresDetectionFast(img, imgVar, imageFile, nome, index): 

    x = y = i =j = k = l = cont = cont1 = chave = aux = pos = nP = nB = None 
    tamLatPor = 20
    w=img.shape[1]
    h=img.shape[0]

    mediaGlobal = numpy.float32(0.0); variancia = None; desvioPadrao = 0.0; mediaOtsu = None; 
    mediaLocal = None; mediaLD=0.0; mediaLE=0.0; mediaLC=0.0; mediaLB=0.0;

    mediaLadosGeral = None; mediaLadosLocal = None; auxFilt = None;
    sumFilterGauss = None; mediaLocalRaio = None

    mediaLDPr=0.0; mediaLEPr=0.0; mediaLCPr=0.0; mediaLBPr=0.0;
    mediaLadosGeralPr = None; mediaLadosLocalPr = None; dist = None; it = None;

    vetPoros = [POROS() for i in range(20000)]
    imgPores = numpy.ones((h, w, 1)) 
    imgB = numpy.ones((h, w, 1)) 

    tamVales = [tamLados() for i in range(w*h)]
    tamCristas = [tamLados() for i in range(w*h)]

    cont = 0

    for y in range(h): 
        for x in range(w): 
            if imgVar[y][x] == 255: 
                mediaGlobal += img[y][x]
                mediaGlobal = float(numpy.format_float_scientific(mediaGlobal, precision=5, exp_digits=2))
                cont += 1 
            
            imgPores[y][x] = 0
            imgB[y][x] = 255

    mediaGlobal = mediaGlobal / cont
    variancia = 0.0

    for y in range(h): 
        for x in range(w): 

            if imgVar[y][x] == 255: 
                variancia += pow(img[y][x] - mediaGlobal, 2)
                variancia = float(numpy.format_float_scientific(variancia, precision=5, exp_digits=2))

    variancia /= cont 
    desvioPadrao = math.sqrt(variancia)

    mediaOtsu = binarizacaoOtsuGlobal(img,imgVar,1);
    mediaGlobal = mediaOtsu;

    x = 0
    while(x < w): 
        aux = 0
        chave = 0

        y = 0
        while(y < h): 
            if(img[y][x] < mediaGlobal):
                if chave == 0:
                    i = y - 1
                    while(i >= aux):    
                        if(y-i <= tamLatPor):
                            tamVales[i * w + x].altB = y-i
                        else: 
                            tamVales[i*w + x].altB = tamLatPor
                        i -= 1        
                aux = y 
                chave = 1

            else: 
                chave = 0
                if(y-aux <= tamLatPor):
                    tamVales[y*w + x].altC = y - aux 
                else: 
                    tamVales[y*w + x].altC = tamLatPor
            y += 1


        if img[h-1][x] > mediaGlobal:
            i = y-1 
            while(i >= aux):

                if(y-i <= tamLatPor): 
                    tamVales[i*w + x].altB = y - i 
                else: 
                    tamVales[i*w + x].altB = tamLatPor

                i -= 1

        x += 1 


    y = 0
    while(y < h): 
        aux = 0
        chave = 0

        x = 0
        while(x < w): 
            if(img[y][x] < mediaGlobal):
                if chave == 0:
                    i = x - 1
                    while(i >= aux):    
                        if(x-i <= tamLatPor):
                            tamVales[y * w + i].largD = x-i
                        else: 
                            tamVales[y*w + i].largD = tamLatPor
                        i -= 1        
                aux = x 
                chave = 1

            else: 
                chave = 0
                if(x-aux <= tamLatPor):
                    tamVales[y*w + x].largE = x - aux 
                else: 
                    tamVales[y*w + x].largE = tamLatPor


            x += 1


        if img[y][w-1] > mediaGlobal:
            i = x-1 
            while(i >= aux):
                if(x-i <= tamLatPor): 
                    tamVales[y*w + i].largD = x - i 
                else: 
                    tamVales[y*w + i].largD = tamLatPor

                i -= 1


        y += 1 

    x = 0
    while(x < w): 
        aux = 0
        chave = 0

        y = 0
        while(y < h): 
            if(img[y][x] > mediaGlobal):
                if chave == 0:
                    i = y - 1
                    while(i >= aux):    
                        if(y-i <= tamLatPor):
                            tamCristas[i * w + x].altB = y-i
                        else: 
                            tamCristas[i*w + x].altB = tamLatPor
                        i -= 1        
                aux = y 
                chave = 1

            else: 
                chave = 0
                if(y-aux <= tamLatPor):
                    tamCristas[y*w + x].altC = y - aux 
                else: 
                    tamCristas[y*w + x].altC = tamLatPor
            y += 1


        if img[h-1][x] < mediaGlobal:
            i = y-1 
            while(i >= aux):

                if(y-i <= tamLatPor): 
                    tamCristas[i*w + x].altB = y - i 
                else: 
                    tamCristas[i*w + x].altB = tamLatPor

                i -= 1

        x += 1 

    y = 0
    while(y < h): 
        aux = 0
        chave = 0

        x = 0
        while(x < w): 
            if(img[y][x] > mediaGlobal):
                if chave == 0:
                    i = x - 1
                    while(i >= aux):    
                        if(x-i <= tamLatPor):
                            tamCristas[y * w + i].largD = x-i
                        else: 
                            tamCristas[y*w + i].largD = tamLatPor
                        i -= 1        
                aux = x 
                chave = 1

            else: 
                chave = 0
                if(x-aux <= tamLatPor):
                    tamCristas[y*w + x].largE = x - aux 
                else: 
                    tamCristas[y*w + x].largE = tamLatPor


            x += 1


        if img[y][w-1] < mediaGlobal:
            i = x-1 
            while(i >= aux):
                if(x-i <= tamLatPor): 
                    tamCristas[y*w + i].largD = x - i 
                else: 
                    tamCristas[y*w + i].largD = tamLatPor

                i -= 1


        y += 1 

    cont = 0; cont1 = 0;
    y = 0 

    while(y < h): 

        x = 0
        while(x < w): 
            if img[y][x] > mediaGlobal and imgVar[y][x] == 255:
                mediaLD += tamVales[y*w+x].largD;
                mediaLE += tamVales[y*w+x].largE;
                mediaLC += tamVales[y*w+x].altC;
                mediaLB += tamVales[y*w+x].altB;
                cont += 1;
            
            if img[y][x] < mediaGlobal and imgVar[y][x] == 255: 
                mediaLDPr += tamCristas[y*w+x].largD;
                mediaLEPr += tamCristas[y*w+x].largE;
                mediaLCPr += tamCristas[y*w+x].altC;
                mediaLBPr += tamCristas[y*w+x].altB;
                cont1 += 1;

            x += 1


        y += 1

    mediaLD /= cont;
    mediaLE /= cont;
    mediaLC /= cont;
    mediaLB /= cont;

    mediaLDPr /= cont1;
    mediaLEPr /= cont1;
    mediaLCPr /= cont1;
    mediaLBPr /= cont1;

    if (mediaLB > tamLatPor):
         mediaLB = tamLatPor;
    if (mediaLC > tamLatPor):
         mediaLC = tamLatPor;
    if (mediaLE > tamLatPor):
         mediaLE = tamLatPor;
    if (mediaLD > tamLatPor):
         mediaLD = tamLatPor;

    if (mediaLBPr > tamLatPor):
         mediaLBPr = tamLatPor;
    if (mediaLCPr > tamLatPor):
        mediaLCPr = tamLatPor;
    if (mediaLEPr > tamLatPor): 
        mediaLEPr = tamLatPor;
    if (mediaLDPr > tamLatPor):
        mediaLDPr = tamLatPor;

    mediaLadosGeral = ((mediaLD + mediaLE + mediaLC + mediaLB)/4);
    mediaLadosGeralPr = ((mediaLDPr + mediaLEPr + mediaLCPr + mediaLBPr)/4);
    mediaGlobal -= 15;

    y = 5

    with tqdm(total = h) as pbar:
        while(y<(h-5)):
            x = 5 
            while(x < (w-5)): 
                
                if img[y][x] > mediaGlobal and imgVar[y][x] == 255: 
                    cont = 0;
                    if (tamVales[y*w+x].largD == tamLatPor): 
                        cont += 1;
                    if (tamVales[y*w+x].largE == tamLatPor): 
                        cont += 1;
                    if (tamVales[y*w+x].altC == tamLatPor):
                        cont += 1;
                    if (tamVales[y*w+x].altB == tamLatPor): 
                        cont += 1;
                    if (cont >= 2): 
                        x += 1
                        continue;

                    cont = 0; cont1 = 0; aux = 0;
                    mediaLadosLocal = 0.0; mediaLadosLocalPr = 0.0;
                    mediaLocal = 0.0;
                    count = 0
                    i =- int(mediaLadosGeralPr*2)
                    while(i <= int(mediaLadosGeralPr*2)):
                        j = -int(mediaLadosGeralPr*2)
                        while(j<=int(mediaLadosGeralPr*2)): 
                            count += 1

                            if (y+i) >=0 and (y+i) < h and (x+j) >= 0 and (x+j) < w and img[y+i][x+j] > mediaGlobal:
                                mediaLadosLocal += (tamVales[(y+i)*w+(x+j)].largD + tamVales[(y+i)*w+(x+j)].largE + tamVales[(y+i)*w+(x+j)].altC + tamVales[(y+i)*w+(x+j)].altB)//4;
                                cont += 1

                            if (y+i) >=0 and (y+i) < h and (x+j) >= 0 and (x+j) < w and img[y+i][x+j] < mediaGlobal:
                                mediaLadosLocalPr += (tamCristas[(y+i)*w+(x+j)].largD + tamCristas[(y+i)*w+(x+j)].largE + tamCristas[(y+i)*w+(x+j)].altC + tamCristas[(y+i)*w+(x+j)].altB)//4;
                                cont1 += 1

                            if (y+i) >=0 and (y+i) < h and (x+j) >=0 and (x+j) < w:
                                mediaLocal += img[y+i][x+j];
                                aux +=1

                            j += 1


                        i += 1

                    numpy.seterr(divide='ignore', invalid='ignore')
                    
                    mediaLadosLocal = numpy.float32(mediaLadosLocal) / cont;
                    mediaLadosLocalPr = numpy.float32(mediaLadosLocalPr) / cont1; 
                    mediaLocal = numpy.float32(mediaLocal) / aux;

                    

                    if (mediaLadosLocal > mediaLadosGeral):
                        mediaLadosLocal = mediaLadosGeral;

                    if (mediaLadosLocalPr > mediaLadosGeralPr):
                        mediaLadosLocalPr = mediaLadosGeralPr;

                    mediaLocalRaio = 0.0;
                    cont = 0; chave = 0; nP = 0; nB = 0;

                    j = numpy.int32(numpy.round(mediaLadosLocalPr/2)-1);

                    if (j > 20):
                        j = 20;
                    if (j < 1):
                        j = 1;
                    while y-j < 0 or y+j > h or x-j < 0 or x+j > w:
                        j -= 1

                    k = tamRaio[j]
                    i = 0
                    while(i<k):
                        l = img[y+RY[j][i]][x+RX[j][i]]
                        mediaLocalRaio += l;

                        if (l > mediaLocal):
                            if (chave == 0):
                                cont += 1
                            chave = 1;
                            nB += 1;
                        else:
                            chave = 0;
                            nP += 1

                        i += 1
                    i=0
                    l = img[y+RY[j][i]][x+RX[j][i]]
                    if (l > mediaLocal):
                        i=k-1;
                        l = img[y+RY[j][i]][x+RX[j][i]]
                        if (l > mediaLocal):
                            cont -= 1;

                    mediaLocalRaio /= k 

                    if (nB > k*0.33):
                        x += 1
                        continue;

                    if (cont >= 2):
                        x += 1
                        continue;

                    if (mediaLocalRaio <= mediaLocal): 
                        imgPores[y][x] = 255

                x += 1

            y += 1
            pbar.update(1)

    boxes = torch.tensor([])
    scores = torch.tensor([])

    nmsWindow = 17


    for i in range(len(imgPores)): 
        for j in range(len(imgPores[0])): 
            if imgPores[i][j] == 255:
                boxes = torch.cat((boxes, torch.tensor([[i, j, i+nmsWindow, j+nmsWindow]])))
                scores = torch.cat((scores, torch.tensor(numpy.array([imgPores[i][j]]))))

    scores.flatten()

    indices = torchvision.ops.boxes.nms(boxes.to(dtype=torch.float64), scores.flatten().to(dtype=torch.float64), 0.2)

    pred = torch.zeros(imgPores.shape)


    for i in indices: 
        x1, y1, x2, y2 = boxes[i]
        center = (x1, y1)
        pred[int(center[0])][int(center[1])] = 255


    imgPores = pred

    # cv2.imwrite("imgRN1.jpeg", img)

    cv2.imwrite(imageFile[0], imgPores.cpu().detach().numpy())

    posX, posY = None, None 

    aux = 0 

    y = 1
    while(y < (h-1)): 

        x = 1
        while(x<(w-1)):

            if imgPores[y][x] == 255: 
                posY = [0]; posX = [0]; cont = [0];
                cont = centroGeomPoro(imgPores,y,x,posY,posX);
                posY[0] = posY[0] // cont;
                posX[0] = posX[0] // cont;
                vetPoros[aux].y = posY[0];
                vetPoros[aux].x = posX[0];
                aux += 1
            
            x += 1

        y += 1


    
    with open(nome, 'w') as f:
        for y in range(aux):
            f.write(str(vetPoros[y].y))
            f.write(", ")
            f.write(str(vetPoros[y].x))
            f.write('\n')





def mainRun(inputImage = "sampleImages/img.jpeg", outputImage = "outputImages/imgRN.jpeg",documentFile = "outputDoc/doc.txt", i=1): 

    img = cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE)
    mask = numpy.ones(img.shape) * 255 

    poresDetectionFast(img, mask, outputImage, documentFile, i)



if __name__=="__main__": 

    parser = argparse.ArgumentParser()


    parser.add_argument('--testingRange', 
                    default=range(1, 2),
                    type=parseNumList, 
                    help="range of data set files that will be used for testing"
                    )    

    parser.add_argument('--groundTruthFolder', 
                    required=True,
                    type=str, 
                    help="Directory where the ground truth dataset is stored"
                    )  

    parser.add_argument('--experimentPath', 
                        required=True,
                        type=str,
                        help="directory where experiment information will be stored"
                        )

    args = parser.parse_args()


    for i in (args.testingRange): 
        print(f"Start working on file #{i}")
        imagePath = args.groundTruthFolder + "/PoreGroundTruthSampleimage/" + str(i) + ".bmp" 
        predictionPath = args.experimentPath + "/Prediction"
        coordinatesPath = predictionPath + "/Coordinates/" + str(i) + ".txt"
        fingeprintPath = predictionPath + "/Fingerprint"
        porePath = predictionPath + "/Pore/" +  str(i) + ".bmp", 

        mainRun(imagePath, porePath, coordinatesPath, i)
        print(imagePath, "complete")

    
    stats = [validate.test(index, groundTruthCoordinatesFolder=args.groundTruthFolder + "/PoreGroundTruthMarked/", predictionCoordinatesFolder=args.experimentPath + "/Prediction/Coordinates/", imageXDimension=cv2.imread(args.groundTruthFolder + "/PoreGroundTruthSampleimage/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE).shape[0], imageYDimension=cv2.imread(args.groundTruthFolder + "/PoreGroundTruthSampleimage/" + str(index) + ".bmp", cv2.IMREAD_GRAYSCALE).shape[1], windowsSize=17, firstIndex=index) for index in args.testingRange]


    sumTrueDetections = sum(i[0] for i in stats) 
    sumFalseDetections = sum(i[1] for i in stats)
    sumPredictions = sum(i[2] for i in stats)
    sumGT = sum(i[3] for i in stats)

    try:
        precision = sumTrueDetections / sumPredictions 
    except ZeroDivisionError: 
        precision = 0

    try:
        recall = sumTrueDetections / sumGT
    except ZeroDivisionError: 
        recall = 0


    try: 
        f_score = 2 * (precision*recall) / (precision+recall)
    except: 
        f_score = 0

    print("F score", str(f_score))
    print("True Detection Rate:", precision)
    print("False Detection Rate:", 1.00-recall)

    

    


    # mainRun()