#ACO Pipeline Part 1
#ACO sorting 

# written by Sarah Betti 2020


import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.simbad import Simbad

import matplotlib.pyplot as plt

import glob
import os


def filesorter(filename, foldername, fitskeyword_to_check, keyword):
    ''' 
    filesorter: moves files to folders based on keywords in headers
    
    INPUTS:
        filename: str - path and name of file
        foldername: str - path and folder to move file into
        fitskeyword_to_check: str - keyword to check in header to move files
        keyword: str - keyword in header 
    '''
    if os.path.exists(filename):
        header = fits.getheader(filename)
        fits_type = header[keyword]

        if os.path.exists(foldername):
            pass
        else:
            os.mkdir(foldername)

        if fits_type == fitskeyword_to_check:
            destination = foldername + '/'
            name = filename.split('/')[-1]
            os.rename(filename, destination + name)  

def remove_bad_frames(rawpath, bad_frame_numbers):
    '''
    removes bad frames from observing night and moves them to a folder called bad_frames
    
    INPUTS:
        rawpath: str - path to raw data folder
        bad_frame_numbers: str or float, optional 
            str: list of bad frames given by file index.  can be given with dashes or commas
                ex: bad_frame_numbers = '1-2,5,8-10'
            float: number of bad frame
    
    '''
    if bad_frame_numbers == None:
        pass
    else:
        # select all files 
        all_fils = np.sort(glob.glob(rawpath + '/*.fit*'))
        # create bad_frames folder
        if os.path.exists(rawpath + '/bad_frames'):
            pass
        else:
            os.mkdir(rawpath + '/bad_frames')

        # if bad_frame_numbers is a str of files, go through them and put them into a list as integers 
        if isinstance(bad_frame_numbers, str):
            fin_nums = []
            nums = bad_frame_numbers.split(',')
            for comma_split_nums in nums:
                if '-' in comma_split_nums:
                    bfn_min, bfn_max =  int(comma_split_nums.split('-')[0]), int(comma_split_nums.split('-')[1])+1
                    bad_frame_list = np.arange(bfn_min, bfn_max)
                    for Q in bad_frame_list:
                        fin_nums.append(Q)
                else:
                    fin_nums.append(int(comma_split_nums))
            bad_frame_numbers = np.array(fin_nums)

        # put integer into list 
        if isinstance(bad_frame_numbers, float):
            bad_frame_numbers = [bad_frame_numbers]
        print('bad frames:', bad_frame_numbers)

        # got through all files and find bad frames.  put them into a folder called bad_frames
        all_fils_nums = np.array([int(i.split('.')[1]) for i in all_fils])
        for i in bad_frame_numbers:
            ind = np.where(all_fils_nums == i)[0]
            if len(ind) != 0:
                name = all_fils[ind[0]].split('/')[-1]
                os.rename(all_fils[ind[0]], rawpath + '/bad_frames/' + name)  
                
def run_filesort(rawpath, targets, bad_frame_numbers=None):
    '''
    Filesort
    
    INPUTS:
    rawpath: str - path to raw data folder
    targets: list - list of target names as they appear in filename
    bad_frame_numbers: str or float, optional 
        str: list of bad frames given by file index.  can be given with dashes or commas
            ex: bad_frame_numbers = '1-2,5,8-10'
        float: number of bad frame
    
    
    '''
    
    # run remove bad frames 
    print('removing bad frames:')
    remove_bad_frames(rawpath, bad_frame_numbers)

    # grab all files that are good 
    print('sort calibration')
    new_all_fils = np.sort(glob.glob(rawpath + '/*.fit*'))
    # sort calibrations
    for fitsfile in new_all_fils:
        filesorter(fitsfile, rawpath+'/flats', 'Flat Field', 'IMAGETYP')
        filesorter(fitsfile, rawpath+'/bias',  'Bias Frame', 'IMAGETYP')
        filesorter(fitsfile, rawpath+'/darks', 'Dark Frame', 'IMAGETYP')
      
    # sort targets
    for i in targets:
        print('sort ',i)
        # grab all targets with target name in file name 
        targ_fils = np.sort(glob.glob(rawpath + '/*' + i + '*.fit*'))
        if len(targ_fils) != 0:
            # remove _ in name 
            target_name = i.replace('_', '')
            # sort targets 
            for fil in targ_fils:
                filesorter(fil, rawpath+'/'+target_name, i.replace('_', ' '), 'OBJECT')
        
        else:
            # if target name is not in the actually filepath, sort by RA/DEC based on querying the target from simbad.  
            
            # grab all targets 
            targ_fils_all = np.sort(glob.glob(rawpath + '/*.fit*'))
            # go through each one
            for fil in targ_fils_all:
                header = fits.getheader(fil)
                #get its RA/DEC
                RA = header['OBJCTRA']
                Dec = header['OBJCTDEC']
                C = SkyCoord(RA, Dec, frame='icrs',unit=(u.hourangle, u.deg))
                
                # search for target in simbad 
                result_table = Simbad.query_object(i)
                Corig = SkyCoord(result_table['RA'][0], result_table['DEC'][0], frame='icrs',unit=(u.hourangle, u.deg))

                # determine how far separated simbad RA/DEC is from frame 
                sep_angle = C.separation(Corig).deg
                #if it is within a degree, move to target folder 
                if sep_angle < 1:
                    if os.path.exists(rawpath + '/'+i):
                        pass
                    else:
                        os.mkdir(rawpath+'/'+i)
                    
                    filename = fil.split('/')[-1]
                    os.rename(fil, rawpath+'/' +i + '/' + filename)  

                   
    