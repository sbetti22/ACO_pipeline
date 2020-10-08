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


#def dark_subtract(filename, path_to_dark, outpath):
#    targetdata = fits.getdata(filename)
#    target_header = fits.getheader(filename)
#    darkdata = fits.getdata(path_to_dark)
#    
#    ds_data = targetdata - darkdata
#    
#    fitsname = filename.split('/')[-1]
#    
#    fits.writeto(outpath + '/d' + fitsname, ds_data, target_header, overwrite=True)
#    return 


def norm_combine_flats(filelist):
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

if not os.path.exists(f'{fil}/Masters'):
    os.makedirs(f'{fil}/Masters')
    print(f'Masters folder created at: {fil}/Masters')
    
masterbiaspath = fil + '/Masters/MasterBias.fit'
if not os.path.exists(f'{fil}/Masters/MasterBias.fit'):
    print('Making Master Bias')
    ### create master bias
    bias_fits = glob.glob(fil + '/bias/*.fit')
    
    median_bias = mediancombine(bias_fits)
    fits.writeto(masterbiaspath, median_bias, header=fits.getheader(bias_fits[0]), overwrite=True)
else:
    print(f'Master Bias in: {masterbiaspath}')


masterdarkpath = fil + '/Masters/'

darkmaster_test = glob.glob(f'{fil}/Masters/MasterDark*.fit')
if len(darkmaster_test) == 0:
    print('Making Master Darks')
    ### create master dark
    dark_outpath = fil + '/darks'
    dark_fits = glob.glob(fil + '/darks/*.fit')
    

    #### bias subtract darks
    for darks in dark_fits:
        bias_subtract(darks, masterbiaspath, dark_outpath)
    #    

    ### median combine bias subtracted dark frames with same exposure time
    b_dark_fits = glob.glob(fil + '/darks/b_*.fit')

    #### sort darks into folders based on exposure time
    for b_darks in b_dark_fits:
        exptime = fits.getheader(b_darks)['EXPTIME']
        filname = b_darks.split('/')[-1]

        if not os.path.exists(fil + '/darks/darks' + str(exptime)):
            os.makedirs(fil + '/darks/darks' +  str(exptime))

        if not os.path.exists(fil + '/darks/darks' + str(exptime) + '/' + filname):
            os.rename(b_darks,fil + '/darks/darks' +  str(exptime) + '/' + filname)

    ### glob all folders 
    b_dark_exptime_folder = glob.glob(fil + '/darks/darks*')

    for exp_folder in b_dark_exptime_folder:
        dark_time = exp_folder.split('/')[-1]
        time = dark_time.split('s')[-1]
        print(f'exposure time {time}')
        b_dark_exptime_fits = glob.glob(exp_folder + '/*.fit')
        median_dark_exptime = mediancombine(b_dark_exptime_fits)
        print('path to dark: ' + masterdarkpath + 'MasterDark' + time + '.fit')
        fits.writeto(masterdarkpath + 'MasterDark' + time + '.fit', median_dark_exptime, header=fits.getheader(b_dark_exptime_fits[0]), overwrite=True)
else:
    print(f'Master Darks: {darkmaster_test}')

flatfield_test = glob.glob(f'{fil}/Masters/MasterFlat*.fits')
if len(flatfield_test) == 0:
    print('Starting Flat fields')
    ### flat fielding

    ### bias subtract flat fields
    print('Starting Bias subtract')
    flat_path = fil + '/flats'
    #
    flat_bands = glob.glob(flat_path + '/*')
    #
    for flat_outpath in flat_bands:
        print(flat_outpath)
        flat_files = glob.glob(flat_outpath + '/*.fit')
        for flats in flat_files:
            bias_subtract(flats, masterbiaspath, flat_outpath)

    print('Starting Dark Subtract')
    ### dark subtract flat fields

    for flat_outpath in flat_bands:
        print(flat_outpath)
        b_flat_files = glob.glob(flat_outpath + '/b_*.fit')

        for b_flats in b_flat_files:
            exptime = fits.getheader(b_flats)['EXPTIME']
            print(exptime)
            masterdark = glob.glob( masterdarkpath + 'MasterDark*.fit')[-1]
#            masterdark = masterdarkpath + 'MasterDark' + str(exptime) + '.fit'
#            masterdark = masterdarkpath + 'MasterDark60.0.fit'
            print(flat_outpath + '/flat' + str(exptime))
            if not os.path.exists(flat_outpath + '/flat' + str(exptime)):
                os.makedirs(flat_outpath + '/flat' + str(exptime))

            dark_subtract(b_flats, masterdark, flat_outpath + '/flat' + str(exptime) )


    ### norm combine flat fields

    for band in flat_bands:
        print(band)
        db_flats = glob.glob(band + '/flat*/*.fit')

        print(db_flats)
        norm_flat = norm_combine_flats(db_flats)

        flat_header = fits.getheader(db_flats[0])
        hdr_band = flat_header['FILTER']
        if hdr_band == 'Red':
            band_name = 'R'
        elif hdr_band == 'Visual':
            band_name = 'V'
        else:
            band_name = 'B'

        print(band_name)
        print(fil + '/Masters/MasterFlat_' + band_name  + '.fit')
        fits.writeto(fil + '/Masters/MasterFlat_' + band_name  + '.fit', norm_flat, flat_header, overwrite=True)
else:
    print(f'Path to Master Flats: {flatfield_test}')









