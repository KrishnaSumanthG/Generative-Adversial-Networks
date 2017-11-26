from __future__ import print_function,division
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import glob

def get_session():
	config=tf.ConfigProto()
	config.gpu_options.allow_growth=True
	session=tf.Session(config=config)
	return session

path="/home/sumanth/Desktop/igl_test"

with get_session() as sess:
	os.chdir(path)
	for i in range(128):
		file1=str(i+1)+"-inputs.png"
		img1=scipy.misc.imread(file1)
		file2=str(i+1)+"-outputs.png"
		img2=scipy.misc.imread(file2)
		file3=str(i+1)+"-targets.png"
		img3=scipy.misc.imread(file3)

		name1="i-"+str(i+1)+".png"
		name2="o-"+str(i+1)+".png"
		name3="t-"+str(i+1)+".png"

		scipy.misc.imsave(name1,img1)
		scipy.misc.imsave(name2,img2)
		scipy.misc.imsave(name3,img3)