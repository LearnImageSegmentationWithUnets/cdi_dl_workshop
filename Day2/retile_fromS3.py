## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

##> Part of a series of notebooks for image recognition and classification using deep convolutional neural networks


#general
from __future__ import division
from glob import glob
import numpy as np 
from imageio import imread, imwrite
from scipy.io import loadmat
import sys, getopt, os

from tile_utils import *

from scipy.stats import mode as md

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

   script, direc, tile, thres, thin = sys.argv    
    
            		 
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
   print(direc) 

   #=======================================================
   outpath = 'S3data_tile_'+str(tile)
   files = sorted([f for f in fs.ls(direc) if f.endswith('.mat')])

   with fs.open(files[0]) as f:
     labels = loadmat(f)['labels']    
    
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

      with fs.open(f) as fid:
         dat = loadmat(fid)         
      labels = dat['labels']

      labels = [label.replace(' ','') for label in labels]	  	  
	  
      res = dat['class']
      del dat

      fim = os.path.normpath(direc+os.sep+f.split(os.sep)[-1].replace('_mres.mat','.JPG'))
      print(fim)  
      if fim: 

         with fs.open(fim, 'rb') as fim:
            image = imread(fim, 'jpg')        
         Z,ind = sliding_window(image, (tile,tile,3), (int(tile/2), int(tile/2),3)) 

         C,ind = sliding_window(res, (tile,tile), (int(tile/2), int(tile/2))) 

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
  

