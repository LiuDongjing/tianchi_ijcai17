import os, sys
tf_python = 'D:/Software/Python35/python.exe'
exec_file = 'tensorflow_main.py'
if len(sys.argv) > 1:
	exec_file = sys.argv[1]
os.system('%s %s'%(tf_python, exec_file))