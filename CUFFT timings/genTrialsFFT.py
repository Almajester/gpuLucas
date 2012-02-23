#
# genTrialsFFT.py to create some trial data for timing different sized CUFFT
#   runlengths for D2Z FFTs with 2**a*3**b*5**c*7**d runlengths

import os

trials = [[a, b, c, d] for a in range(0, 30)
          for b in range(0, 25) for c in range(0, 20) for d in range(0, 15)]
trials2 = [ x for x in trials if 2**18 <= 2**x[0]*3**x[1]*5**x[2]*7**x[3] < 2**24]

trials3 = [[2**x[0]*3**x[1]*5**x[2]*7**x[3]] + x for x in trials2]

#sort by size
trials4 = sorted(trials3, key=lambda trial: trial[0])

commandStr = "./gpuLucasFFTcarrytrials "

for i in trials4:
	mycommand = commandStr + "-Z 1 " + str(i[1]) + " " + str(i[2]) + " " + str(i[3]) + " " + str(i[4])
	os.system(mycommand)
