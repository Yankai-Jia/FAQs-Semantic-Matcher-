import numpy
import os

os.chdir('/Users/newworld/Documents/UTDCOURSES/6375/Project/libsvm-3.22/python')
import sys
sys.path.append('/Users/newworld/Documents/UTDCOURSES/6375/Project/libsvm-3.22/python')
from svmutil import *
y,x=svm_read_problem('libsvm_train_nlp_data_1.txt')
yt,xt=svm_read_problem('libsvm_test_nlp_data_1.txt')
model=svm_train(y,x)
p_label, p_acc, p_val = svm_predict(yt, xt, model)
print(p_label)
