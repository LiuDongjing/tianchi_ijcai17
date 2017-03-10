import os
tboard = 'D:/Software/Python35/Scripts/tensorboard.exe'
log_dir = '../temp/tf_log'
os.system('%s --logdir=%s'%(tboard, log_dir))