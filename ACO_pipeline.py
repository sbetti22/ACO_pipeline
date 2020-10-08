# ACO pipeline

# written by Sarah Betti 2020

from sort_filter import *
from reduction_pipeline import *
from photometry_pipeline import *
from clean_up import *

########### 0 #############

# path to raw data
fil = '/Users/sarah/Downloads/20201006'
# target names in filename 
sort_target_names = ['M_39', 'FS150']
# filters
filters = ['B', 'V', 'R']
# cluster name
clustername = 'M39'
# standard star name 
standardname='FS150'
# bad frame numbers
bad_frame_numbers = '1-10,21-22,29-39,41-45,55,57,68,70-75'

# standard star position for register and cross-correlation
centerx = 2570
centery = 1930

# standard star magnitudes
B_mag = '11.53 [0.07]'
V_mag='11.07 [0.07]'
R_mag ='10.80 [0.09]'

####################################

# targets
targs = [clustername, standardname]

# sort files  
run_filesort(fil, sort_target_names, bad_frame_numbers=bad_frame_numbers)

# do calibration 
run_master_bias(fil)
run_master_dark(fil)
run_master_flat(fil)
run_targets(fil,targs)


# flip images over meridian - only use if telescope flipped meridian.  
flip_R_M39 = glob.glob(fil + '/M39/Rband/fdb*.fit')
flip_R_FS150 = glob.glob(fil + '/FS150/Rband/fdb*.fit')

flip_images(flip_R_M39)
flip_images(flip_R_FS150)


# register images
run_register_align(fil, standardname, filters, centerx=centerx, centery=centery, boxsize=1400)

run_register_align(fil, clustername, filters, centerx=None, centery=None, boxsize=500)

# create RGB image
run_RGB(fil, clustername, filters,siglowhi = [-1,20.,-1,20.,-1,20.])


# do photometry 
run_photometry(fil, clustername, standardname, plot=True)

calibrate_photometry(fil,clustername, standardname, plot=True, 
                     B_mag = B_mag, V_mag=V_mag, R_mag =R_mag)

 plot CMD
plot_CMD(fil, clustername, 'B', 'V', save=True)

 clean up files
clean_up(fil, targs)


