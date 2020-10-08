# ACO pipeline Part 2
# ACO reduction pipeline

# written by Sarah Betti 2020
# bias_subtract, dark_subtract,, norm_combine_flats, mediancombine, centroid, cross_image, shift_images, scale_filter, run_RGB function written by Kim Ward-Duong 2018-2019

import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import glob
import os
import time

import astropy
from astropy.io import fits
from astropy import stats
from astropy.stats import sigma_clip

import scipy.signal

from PIL import Image


def mediancombine(filelist):
    '''
    median combine frames
    '''
    n = len(filelist)
    first_frame_data = fits.getdata(filelist[0])
    imsize_y, imsize_x = first_frame_data.shape
    fits_stack = np.zeros((imsize_y, imsize_x , n))
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        fits_stack[:,:,ii] = im
    med_frame = np.median(fits_stack, axis = 2)
    return med_frame

def bias_subtract(filename, path_to_bias, outpath):
    '''
    bias subtract frames
    '''
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
    normalize and combine frames
    '''
    n = len(filelist)
    first_frame_data = fits.getdata(filelist[0])
    imsize_y, imsize_x = first_frame_data.shape
    fits_stack = np.zeros((imsize_y, imsize_x , n))
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        norm_im = im/np.median(im) 
        fits_stack[:,:,ii] = norm_im

    med_frame = np.median(fits_stack, axis=2)
    return med_frame

def centroid(data_arr,xcen,ycen,nhalf=5,derivshift=1.):
    '''
    centroid
    -----------------
    based on dimension-indepdendent line minimization algorithms implemented in IDL cntrd.pro 
    
    inputs
    ----------------
    data_arr      : (matrix of floats) input image
    xcen          : (int) input x-center guess
    ycen          : (int) input y-center guess
    nhalf         : (int, default=5) the excised box of pixels to use
                    recommended to be ~(2/3) FWHM (e.g. only include star pixels).
    derivshift    : (int, default=1) degree of shift used to calculate derivative. 
                     larger values can find shallower slopes more efficiently
    
    
    
    outputs
    ---------------
    xcenf         : the centroided x value
    ycenf         : the centroided y value
    
    dependencies
    ---------------
    numpy         : imported as np
    
    also see another implementation here:
    https://github.com/djones1040/PythonPhot/blob/master/PythonPhot/cntrd.py
    
    function written by: Dr. Kimberly Ward-Duong
    
    '''
    # input image requires the transpose to 
    #
    # find the maximum value near the given point
    data = data_arr[int(ycen-nhalf):int(ycen+nhalf+1),int(xcen-nhalf):int(xcen+nhalf+1)]


    yadjust = nhalf - np.where(data == np.max(data))[0][0]
    xadjust = nhalf - np.where(data == np.max(data))[1][0]
    
    xcen -= xadjust
    ycen -= yadjust
    
    # now use the adjusted centers to find a better square
    data = data_arr[int(ycen-nhalf):int(ycen+nhalf+1),int(xcen-nhalf):int(xcen+nhalf+1)]

    # make a weighting function
    ir = (nhalf-1) > 1 
    
    # sampling abscissa: centers of bins along each of X and Y axes
    nbox = 2*nhalf + 1
    dd = np.arange(nbox-1).astype(int) + 0.5 - nhalf
    
    #Weighting factor W unity in center, 0.5 at end, and linear in between 
    w = 1. - 0.5*(np.abs(dd)-0.5)/(nhalf-0.5) 
    sumc   = np.sum(w)
    
    # fancy comp sci part to find the local maximum
    #
    # this uses line minimization using derivatives
    # (see text such as Press' Numerical Recipes Chapter 10), 
    # treating X and Y dimensions as indepdendent (generally safe for stars). 
    # In this sense the method can be thought of as a two-step gradient descent.

    # find X centroid
    # shift in Y and subtract to get derivative
    deriv = np.roll(data,-1,axis=1) - data.astype(float)
    deriv = deriv[nhalf-ir:nhalf+ir+1,0:nbox-1]
    deriv = np.sum( deriv, 0 )                    #    ;Sum X derivatives over Y direction

    sumd   = np.sum( w*deriv )
    sumxd  = np.sum( w*dd*deriv )
    sumxsq = np.sum( w*dd**2 )
    
    dx = sumxsq*sumd/(sumc*sumxd)
    
    xcenf = xcen - dx

    # find Y centroid
    # shift in X and subtract to get derivative
    deriv = np.roll(data,-1,axis=0) - data.astype(float)    # Shift in X & subtract to get derivative
    deriv = deriv[0:nbox-1,nhalf-ir:nhalf+ir+1]
    deriv = np.sum( deriv,1 )               #    ;Sum X derivatives over Y direction

    sumd   = np.sum( w*deriv )
    sumxd  = np.sum( w*dd*deriv )
    sumxsq = np.sum( w*dd**2 )
    
    dy = sumxsq*sumd/(sumc*sumxd)
    
    ycenf = ycen - dy
    
    return xcenf,ycenf

def cross_image(im1, im2, xcen, ycen, boxsize):
    '''
    cross_image
    ---------------
    calcuate cross-correlation of two images in order to find shifts
    
    
    inputs
    ---------------
    im1                      : (matrix of floats)  first input image
    im2                      : (matrix of floats) second input image
    boxsize                  : (integer, optional) subregion of image to cross-correlate
    
    
    returns
    ---------------
    xshift                   : (float) x-shift in pixels
    yshift                   : (float) y-shift in pixels
    
    dependencies
    ---------------
    scipy.signal.fftconvolve : two-dimensional fourier convolution
    centroid                 : a centroiding algorithm of your choosing or defintion
    numpy                    : imported as np
    
    todo
    ---------------
    -add more **kwargs capabilities for centroid argument
    
    '''
    
    # The type cast into 'float' is to avoid overflows:
    im1_gray = im1.astype('float')
    im2_gray = im2.astype('float')

    # Enable a trimming capability using keyword argument option.
    
    im1_gray = im1_gray[ycen-boxsize:ycen+boxsize,xcen-boxsize:xcen+boxsize]
    im2_gray = im2_gray[ycen-boxsize:ycen+boxsize,xcen-boxsize:xcen+boxsize]

    # Subtract the averages (means) of im1_gray and im2_gray from their respective arrays     
    im1_gray -= np.nanmean(im1_gray)
    im2_gray -= np.nanmean(im2_gray)
    
    # guard against extra nan values
    im1_gray[np.isnan(im1_gray)] = np.nanmedian(im1_gray)
    im2_gray[np.isnan(im2_gray)] = np.nanmedian(im2_gray)


    # Calculate the correlation image using fast Fourrier Transform (FFT)
    # Note the flipping of one of the images (the [::-1]) to act as a high-pass filter
    corr_image = scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')

    # Find the peak signal position in the cross-correlation, which gives the shift between the images
    corr_tuple = np.unravel_index(np.nanargmax(corr_image), corr_image.shape)
    
    try: # try to use a centroiding algoritm to find a better peak
        xcenc,ycenc = centroid(corr_image.T,corr_tuple[0],corr_tuple[1],nhalf=10,derivshift=1.)

    except: # if centroiding algorithm fails, use just peak pixel
        xcenc,ycenc = corr_tuple
        
    # Calculate shifts (distance from central pixel of cross-correlated image)
    xshift = xcenc - corr_image.shape[0]/2.
    yshift = ycenc - corr_image.shape[1]/2.

    return xshift,yshift

def shift_image(image,xshift,yshift):
    '''
    shift_image
    -------------
    wrapper for scipy's implementation that shifts images according to values from cross_image
    
    inputs
    ------------
    image           : (matrix of floats) image to be shifted
    xshift          : (float) x-shift in pixels
    yshift          : (float) y-shift in pixels
    
    outputs
    ------------
    shifted image   : shifted, interpolated image. 
                      same shape as input image, with zeros filled where the image is rolled over
    '''
    return scipy.ndimage.interpolation.shift(image,(xshift,yshift))

def scale_filter(tmpimg,lowsig,highsig):
    '''
    scale images for combining into RGB image
    '''

    tmpimg -= np.median(tmpimg)
    print('minmax 1: ', np.min(tmpimg),np.max(tmpimg))


    tmpsig = stats.sigma_clipped_stats(tmpimg, sigma=2, maxiters=5)[2]
    print('std: ', tmpsig)
    print("lowsig, highsig: ", lowsig, highsig)
    print('cuts: ', lowsig*tmpsig, highsig*tmpsig)

#    image_hist = plt.hist(tmpimg.flatten(), 1000, range=[-100,100])

    # apply thresholding
    tmpimg[np.where(tmpimg < lowsig*tmpsig)] = lowsig*tmpsig
    tmpimg[np.where(tmpimg > highsig*tmpsig)] = highsig*tmpsig
    print('minmax 2: ', np.min(tmpimg),np.max(tmpimg))

    # double hyperbolic arcsin scaling
    tmpimg = np.arcsinh(tmpimg)
    print('minmax 3: ', np.min(tmpimg),np.max(tmpimg))

    # scale to [0,255]
    tmpimg += np.min(tmpimg)
    tmpimg *= 255./np.max(tmpimg)
    tmpimg[np.where(tmpimg < 0.)] = 0.
    print('minmax 4: ', np.min(tmpimg),np.max(tmpimg))
    
    # recast as unsigned integers for jpeg writer
    IMG = Image.fromarray(np.uint8(tmpimg))
    
    print("")
    
    return IMG

####################################

def run_master_bias(fil):
    '''
    create master bias
    '''
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
    print()

def run_master_dark(fil):
    '''
    create master darks for each exposure time
    '''
    masterbiaspath = fil + '/Masters/MasterBias.fit'
    masterdarkpath = fil + '/Masters/'

    darkmaster_test = glob.glob(f'{fil}/Masters/MasterDark*.fit')
    if len(darkmaster_test) == 0:
        print('Making Master Darks')
        ### create master dark
        dark_outpath = fil + '/darks'
        b_dark_test = glob.glob(fil + '/darks/b_*.fit')
        for im in b_dark_test:
            os.remove(im)
        dark_fits = glob.glob(fil + '/darks/*.fit')

        #### bias subtract darks
        for darks in dark_fits:
            bias_subtract(darks, masterbiaspath, dark_outpath)

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
    print()
        
def run_master_flat(fil):
    '''
    create master flats for each filter
    '''
    masterbiaspath = fil + '/Masters/MasterBias.fit'
    masterdarkpath = fil + '/Masters/'
    flatfield_test = glob.glob(f'{fil}/Masters/MasterFlat*.fits')
    if len(flatfield_test) == 0:
        print('Starting Flat fields')

        ### bias subtract flat fields
        b_flat_test = glob.glob(fil + '/flats/b_*.fit')
        for im in b_flat_test:
            os.remove(im)        

        flat_files = glob.glob(fil + '/flats/*.fit')
        for flats in flat_files:
            bias_subtract(flats, masterbiaspath, fil+'/flats')

        ### dark subtract flat fields
        db_flat_test = glob.glob(fil+'/flats/*/db_*.fit')
        for im in db_flat_test:
            os.remove(im)
                
        b_flat_files = glob.glob(fil + '/flats/b_*.fit')

        for b_flats in b_flat_files:
            exptime = fits.getheader(b_flats)['EXPTIME']
            filters = fits.getheader(b_flats)['FILTER'][0]
            if os.path.exists(masterdarkpath + 'MasterDark' + str(exptime) + '.fit'):
                masterdark = masterdarkpath + 'MasterDark' + str(exptime) + '.fit'
            else:
                masterdark = glob.glob( masterdarkpath + 'MasterDark*.fit')[-1]

            if not os.path.exists(fil + '/flats/' + filters + 'flat'):
                os.makedirs(fil + '/flats/' + filters + 'flat')

            dark_subtract(b_flats, masterdark, fil + '/flats/' + filters + 'flat')


        ### norm combine flat fields
        flat_bands = glob.glob(fil + '/flats/*flat')
        for band in flat_bands:
            db_flats = glob.glob(band + '/db_*.fit')

            norm_flat = norm_combine_flats(db_flats)

            flat_header = fits.getheader(db_flats[0])
            band_name = flat_header['FILTER'][0]

            print('path to '+ band_name + ' flat: ' + fil + '/Masters/MasterFlat_' + band_name  + '.fit')
            fits.writeto(fil + '/Masters/MasterFlat_' + band_name  + '.fit', norm_flat, flat_header, overwrite=True)
    else:
        print(f'Path to Master Flats: {flatfield_test}')
    print()
    
def run_targets(fil, targs):
    '''
    bias, dark, and flat field science targets
    '''
    for target in targs:
        print()
        print('------------o------------')
        print('target: ', target)
        print()
        masterbiaspath = fil + '/Masters/MasterBias.fit'
        masterpath = fil + '/Masters/'

        # bias subtract targets
        bias_images = glob.glob(f'{fil}/{target}/b_*.fit')
        scidata = glob.glob(fil + '/' + target + '/*.fit')

        filters = []

        if len(bias_images)!=len(scidata):
            [os.remove(im) for im in bias_images]
            print('Bias subtracting ')
            for sci_image in scidata:
                filtername = fits.getheader(sci_image)['FILTER'][0]          
                sci_outpath = fil + '/' + target + '/' + filtername + 'band'
                if not os.path.exists(sci_outpath):
                    os.makedirs(sci_outpath)
                bias_subtract(sci_image, masterbiaspath, sci_outpath)
                filters.append(filtername)
        filters = np.unique(filters)

        # dark subtract bias targets
        for filtername in filters:
            b_scidata = glob.glob(fil + '/' + target + '/' + filtername + 'band/b_*.fit')

            dark_images = glob.glob(f'{fil}/{target}/{filtername}band/db*.fit')

            if len(dark_images)!=len(b_scidata):
                print('Dark subtracting ', filtername, ' band')
                [os.remove(im) for im in dark_images]
                sci_outpath = fil + '/' + target + '/' + filtername + 'band'
                for b_sci_image in b_scidata:
                    exptime = fits.getheader(b_sci_image)['EXPTIME']
                    if os.path.exists(masterpath + 'MasterDark' + str(exptime) + '.fit'):
                        masterdark = masterpath + 'MasterDark' + str(exptime) + '.fit'
                    else:
                        masterdark = glob.glob( masterpath + 'MasterDark*.fit')[-1]

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
                    fits.writeto(sci_outpath + '/f' + sci_name, fdb_sci_image, db_sci_hdr, overwrite=True )
                    
def flip_images(filenames):
    '''
    rotate images if meridian was flipped
    '''
    for i in filenames:
        im = fits.getdata(i)
        im_flip = np.flip(im, axis=(1,0))
        fits.writeto(i, im_flip, header=fits.getheader(i), overwrite=True)
                  
def run_register_align(datadir, targs, filters, centerx=None, centery=None, boxsize=1400):
    '''
    register images 
    '''
    if isinstance(targs, str):
        targs = [targs]
    if isinstance(centerx, int):
        centerx = [centerx]
        centery = [centery]
    
    print(targs)
    print(np.ones_like(targs).astype(int))
    if centerx==None:
        centerx = np.ones_like(targs).astype(int) * 2048
    if centery==None:
        centery = np.ones_like(targs).astype(int) * 2048
    
    # Cycle through list of targets:
    for ind, targname in enumerate(targs):
        print(' ')
        print('-----------------------------')      
        print('target: ', targname)
        print('-----------------------------')      

        # Using glob, make list of all reduced images of current target in all filters.
        print(datadir + '/' + targname + '/*band/fdb*.fit')
        imlist = glob.glob(datadir + '/' + targname + '/*band/fdb*.fit')

        # Check to make sure that your new list has the right files:
        print("All files to be aligned: \n", imlist)
        print()
        print(len(imlist), ' files to be aligned')
        print('\n') # adding some space to the print statements, '/n' means new line

        # Open first image = master image; all other images of same target will be aligned to this one.
        im1,hdr1 = fits.getdata(imlist[0],header=True)
        print("Aligning all images to:", imlist[0])

        print('\n') # adding some space to the print statement
        

        xshifts = {}
        yshifts = {}

        for index,filename in enumerate(imlist):
            im,hdr = fits.getdata(filename,header=True)
            xshifts[index], yshifts[index] = cross_image(im1, im, centerx[ind], centery[ind], boxsize=boxsize)
            print("Shift for image", index, "is", xshifts[index], yshifts[index])

        # Calculate trim edges of new median stacked images so all stacked images of each target have same size 
        max_x_shift = int(np.max([xshifts[x] for x in xshifts.keys()]))
        max_y_shift = int(np.max([yshifts[x] for x in yshifts.keys()]))

        print('   Max x-shift={0}, max y-shift={1} (pixels)'.format(max_x_shift,max_y_shift))


        # Cycle through list of filters
        for filtername in filters:
            # Create a list of FITS files matching *only* the selected filter:
            scilist = glob.glob(datadir + '/' + targname + '/' + filtername + 'band/fdb*.fit')

            if len(scilist) < 1:
                print("Warning! No files in scilist. Your path is likely incorrect.")
                break

#            # TESTING: Complete the for loop below that ensures that each of the scilist entries has the right filter:
#            for fitsfile in scilist:
#                filt = fits.getheader(fitsfile)['FILTER']
#                print(filt)

            nfiles = len(scilist)
            print('Stacking ', nfiles, filtername, ' science frames')

            # Define new array with same size as master image
            image_stack = np.zeros([im1.shape[0],im1.shape[1],len(scilist)])

            xshifts_filt = {}
            yshifts_filt = {}
            for index,filename in enumerate(scilist):
                im,hdr = fits.getdata(filename,header=True)
                xshifts_filt[index], yshifts_filt[index] = cross_image(im1, im, centerx[ind], centery[ind], boxsize=boxsize)
                image_stack[:,:,index] = shift_image(im,xshifts_filt[index], yshifts_filt[index])

            median_image = np.median(image_stack,axis=2)

#            # Sets the new image boundaries
#            if (max_x_shift > 0) & (max_y_shift > 0): # don't apply cut if no shift!
#                median_image = median_image[max_x_shift:-max_x_shift,max_y_shift:-max_y_shift]

            # Make a new directory in your datadir for the new stacked fits files
            if os.path.isdir(datadir + '/Stacked') == False:
                os.mkdir(datadir + '/Stacked')
                print('\n Making new subdirectory for stacked images:', datadir + '/Stacked \n')

            # Save the final stacked images into your new folder:
            fits.writeto(datadir + '/Stacked/' + targname + '_' + filtername + 'stack.fits', median_image, fits.getheader(scilist[0]), overwrite=True)
            print('   Wrote FITS file ',targname+'_'+filtername+'stack.fits', 'in ',datadir + '/Stacked/','\n')

    print('\n Done stacking!')
    
def run_RGB(datadir, targname, filters, siglowhi):
    '''
    create RGB image
    '''
#    siglowhi = [-2,10.,-5,15.,-2,11.] # BGR low and high sigma limits. Ex: To make red brighter, make 6th # lower.
    
    # Read in 3 images 
    Rtmp = fits.getdata(datadir+'/Stacked/'+targname+'_'+filters[2]+'stack.fits')
    Gtmp = fits.getdata(datadir+'/Stacked/'+targname+'_'+filters[1]+'stack.fits')
    Btmp = fits.getdata(datadir+'/Stacked/'+targname+'_'+filters[0]+'stack.fits')

    # Scale all 3 images
    print('Calculating stats....')
    R = scale_filter(Rtmp,lowsig=siglowhi[4],highsig=siglowhi[5])
    G = scale_filter(Gtmp,lowsig=siglowhi[2],highsig=siglowhi[3])
    B = scale_filter(Btmp,lowsig=siglowhi[0],highsig=siglowhi[1])

    # Merge 3 images into one RGB image
    im = Image.merge("RGB", (R,G,B))

    im.save(datadir+'/Stacked/'+targname+'_RGB.jpg', "JPEG")
    print("Saved image as ", datadir+'/Stacked/'+targname+'_RGB.jpg')