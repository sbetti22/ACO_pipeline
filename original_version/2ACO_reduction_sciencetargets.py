import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import os


def mediancombine(filelist):

    # Add your own comment describing this step
    n = len(filelist)

    # Add your own comment describing this step
    first_frame_data = fits.getdata(filelist[0])

    # Add your own comment describing this step
    imsize_y, imsize_x = first_frame_data.shape

    # Add your own comment describing this step
    fits_stack = np.zeros((imsize_y, imsize_x , n))

    # Add your own comment describing this step
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        fits_stack[:,:,ii] = im

    # Add your own comment describing this step
    med_frame = np.median(fits_stack, axis = 2)

    return med_frame

def bias_subtract(filename, path_to_bias, outpath):
    targetdata = fits.getdata(filename)
    target_header = fits.getheader(filename)
    biasdata = fits.getdata(path_to_bias)

    b_data = targetdata - biasdata

    # check if files exist in directory other than notebook
    fitsname = filename.split('/')[-1]

    fits.writeto(outpath + "/" + 'b_' + fitsname, b_data, target_header, overwrite=True)
    return

def dark_subtract(filename, path_to_dark, outpath):
    '''
    performs dark subtraction on your flat/science fields. 
    '''
    
    # open the flat/science field data and header
    frame_data = fits.getdata(filename)
    frame_header = fits.getheader(filename)
    
    #open the master dark frame with the same exposure time as your data. 
    master_dark_data = fits.getdata(path_to_dark)
    
    #subtract off the dark current 
    if frame_header['EXPTIME'] != fits.getheader(path_to_dark)['EXPTIME']:
        scale = frame_header['EXPTIME'] / fits.getheader(path_to_dark)['EXPTIME']
        master_dark_data = scale * master_dark_data
    
    dark_subtracted = frame_data - master_dark_data
    
    new_filename = filename.split('/')[-1]
    

    fits.writeto(outpath + '/d' + new_filename, dark_subtracted, frame_header,overwrite=True)
    return 


def norm_combine_flats(filelist):
    '''
    Edit this docstring accordingly!
    '''
    # Add your own comment describing this step
    n = len(filelist)

    # Add your own comment describing this step
    first_frame_data = fits.getdata(filelist[0])

    # Add your own comment describing this step
    imsize_y, imsize_x = first_frame_data.shape

    # Add your own comment describing this step
    fits_stack = np.zeros((imsize_y, imsize_x , n))

    # Add your own comment describing this loop
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        norm_im = im/np.median(im) # finish new line here to normalize flats
        fits_stack[:,:,ii] = norm_im

    # Add your own comment describing this step
    med_frame = np.median(fits_stack, axis=2)
    return med_frame
    
    
    

fil = '/Users/sarah/Downloads/20200923'
targs = ['Moon']
filters = ['B', 'V', 'R']

for target in targs:
    print()
    print('------------o------------')
    print('target: ', target)
    print()
    masterbiaspath = fil + '/Masters/MasterBias.fit'
    masterpath = fil + '/Masters/'

    ### bias subtract targets

    for filtername in filters:
        bias_images = glob.glob(f'{fil}/{target}/{filtername}band/b_*.fit')
        scidata = glob.glob(fil + '/' + target + '/' + filtername + 'band/*.fit')

        if len(bias_images)!=len(scidata):
            [os.remove(im) for im in bias_images]
            print('Bias subtracting ', filtername, ' band')
            sci_outpath = fil + '/' + target + '/' + filtername + 'band'
            print(sci_outpath)
            for sci_image in scidata:
                bias_subtract(sci_image, masterbiaspath, sci_outpath)



    ### dark subtract bias targets

    for filtername in filters:
        b_scidata = glob.glob(fil + '/' + target + '/' + filtername + 'band/b_*.fit')

        dark_images = glob.glob(f'{fil}/{target}/{filtername}band/db*.fit')

        if len(dark_images)!=len(b_scidata):
            print('Dark subtracting ', filtername, ' band')
            [os.remove(im) for im in dark_images]
            sci_outpath = fil + '/' + target + '/' + filtername + 'band'
            for b_sci_image in b_scidata:
                exptime = fits.getheader(b_sci_image)['EXPTIME']
                #
                #get master dark for exposure time
                masterdark = glob.glob(masterpath + 'MasterDark*.fit')[-1]
#                masterdark = masterpath + 'MasterDark' + str(exptime) + '.fit'
#                masterdark = masterpath + 'MasterDark60.0.fit'
                dark_subtract(b_sci_image, masterdark, sci_outpath)


    # flat field db targets

    for filtername in filters:
        db_scidata = glob.glob(fil + '/' + target + '/' + filtername + 'band/db_*.fit')
        flat_images = glob.glob(f'{fil}/{target}/{filtername}band/fdb*.fit')

        if len(flat_images)!=len(db_scidata): 
            print('Flat Fielding ', filtername, ' band')
            [os.remove(im) for im in flat_images]
            masterflat = masterpath + '/MasterFlat_' + filtername + '.fit'
            masterflat_data = fits.getdata(masterflat)

            sci_outpath = fil + '/' + target + '/' + filtername + 'band'

            for db_sci_image in db_scidata:
                db_sci_data = fits.getdata(db_sci_image)
                db_sci_hdr = fits.getheader(db_sci_image)

                fdb_sci_image = db_sci_data / masterflat_data 
                sci_name = db_sci_image.split('/')[-1]
                #print(sci_outpath + '/f' + sci_name)
                fits.writeto(sci_outpath + '/f' + sci_name, fdb_sci_image, db_sci_hdr, overwrite=True )



        
        
        
        
        
        
        





