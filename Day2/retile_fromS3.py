## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

#general
from __future__ import division
##from joblib import Parallel, delayed
from glob import glob
import numpy as np 
#from scipy.misc import imread
from imageio import imread, imwrite
from scipy.io import loadmat
import sys, getopt, os

from tile_utils import *

from scipy.stats import mode as md
#from scipy.misc import imsave

#if sys.version[0]=='3':
#   from tkinter import Tk, Toplevel 
#   from tkinter.filedialog import askopenfilename
#   import tkinter
#   import tkinter as tk
#   from tkinter.messagebox import *   
#   from tkinter.filedialog import *
#else:
#   from Tkinter import Tk, TopLevel
#   from tkFileDialog import askopenfilename
#   import Tkinter as tkinter
#   import Tkinter as tk
#   from Tkinter.messagebox import *   
#   from Tkinter.filedialog import *   
   
import os.path as path

import s3fs
fs = s3fs.S3FileSystem(anon=True)

# =========================================================
def writeout(tmp, cl, labels, outpath, thres):

   l, cnt = md(cl.flatten())
   l = np.squeeze(l)
   if cnt/len(cl.flatten()) > thres:
      outfile = id_generator()+'.jpg'
      fp = outpath+os.sep+labels[l]+os.sep+outfile
      imwrite(fp, tmp) ##imsave(fp, tmp)

#==============================================================
if __name__ == '__main__':

   # read the WYSS section for how to run this
   script, direc, tile, thres, thin = sys.argv    
    
   #direc = ''; tile = ''; thres = ''; thin = ''

   #argv = sys.argv[1:]
   #try:
   #   opts, args = getopt.getopt(argv,"hi:t:a:b:")
   #except getopt.GetoptError:
   #   print('python retile.py -i direc -t tilesize -a threshold -b proportion_thin')
   #   sys.exit(2)

   #for opt, arg in opts:
   #   if opt == '-h':
   #      print('Example usage: python retile.py -i direc -t 96 -a 0.9 -b 0.5')
   #      sys.exit()
   #   elif opt in ("-t"):
   #      tile = arg
   #   elif opt in ("-a"):
   #      thres = arg
   #   elif opt in ("-b"):
   #      thin = arg
            		 
   if not direc:
      direc = 'train'
   if not tile:
      tile = 96
   if not thres:
      thres = .9
   if not thin:
      thin = 0
	  
   tile = int(tile)
   thres = float(thres)
   thin = float(thin)

   #===============================================
   # Run main application
   #Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing   
   #files = askopenfilename(filetypes=[("pick mat files","*.mat")], multiple=True)  
    
   #direc = imdirec = os.path.dirname(files[0])##'useimages'
   #direc = 'cdi-workshop/semseg_data/ontario/test'
   print(direc) 

   #=======================================================
   outpath = 'S3data_tile_'+str(tile)
   #files = sorted(glob(direc+os.sep+'*.mat'))
   files = sorted([f for f in fs.ls(direc) if f.endswith('.mat')])

   with fs.open(files[0]) as f:
     labels = loadmat(f)['labels']    
    
   ##labels = loadmat(files[0])['labels']

   labels = [label.replace(' ','') for label in labels]
   #=======================================================
    
   #=======================================================
   try:
      os.mkdir(outpath)
   except:
      pass

   for f in labels:
      try:
         os.mkdir(outpath+os.sep+f)
      except:
         pass
   #=======================================================

   types = ('.jpg', '.jpeg', '.tif', '.tiff', '.png') # the tuple of file types   
   files_grabbed = []
   for t in types:    
      files_grabbed.extend(f for f in fs.ls(direc) if f.endswith(t))
   print(len(files_grabbed))

   #=======================================================
   for f in files:

      ##dat = loadmat(f)
      with fs.open(f) as fid:
         dat = loadmat(fid)         
      labels = dat['labels']

      labels = [label.replace(' ','') for label in labels]	  	  
	  
      res = dat['class']
      del dat
      #core = f.split(os.sep)[-1].split('_mres')[0]  #'/'
      #print(core)
      ##get the file that matches the above pattern but doesn't contain 'mres'	  
      #fim = [e for e in files_grabbed if e.find(core)!=-1 if e.find('mres')==-1 ]
    
      #fim = direc+os.sep+f.split(os.sep)[-1].replace('_mres.mat','.JPG')
      fim = os.path.normpath(direc+os.sep+f.split(os.sep)[-1].replace('_mres.mat','.JPG'))
      print(fim)  
      if fim: 
         #print(fim)   
         ##fim = direc+os.sep+fim ##fim[0]
         #print('Generating tiles from dense class map ....')
        
         with fs.open(fim, 'rb') as fim:
            image = imread(fim, 'jpg')        
         Z,ind = sliding_window(image, (tile,tile,3), (int(tile/2), int(tile/2),3)) 

         C,ind = sliding_window(res, (tile,tile), (int(tile/2), int(tile/2))) 

         ##w = Parallel(n_jobs=-1, verbose=0, pre_dispatch='2 * n_jobs', max_nbytes=None)(delayed(writeout)(Z[k], C[k], labels, outpath, thres) for k in range(len(Z))) 
         for k in range(len(Z)):
            writeout(Z[k], C[k], labels, outpath, thres)
      else:
         print("correspodning image not found")	     
		 
   print('thinning files ...')
   if thin>0:
      for f in labels:   
         files = glob(outpath+os.sep+f+os.sep+'*.jpg')
         if len(files)>60:   
            usefiles = np.random.choice(files, int(thin*len(files)), replace=False)   
            rmfiles = [x for x in files if x not in usefiles.tolist()] 
            for rf in rmfiles:
               os.remove(rf)
	  
   for f in labels:
      files = glob(outpath+os.sep+f+os.sep+'*.jpg')
      print(f+': '+str(len(files)))
  

   

