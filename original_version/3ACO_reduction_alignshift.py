import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy.io import fits
import matplotlib.cm as cm
import scipy.signal
import glob
import os
from astropy.stats import sigma_clip
import time

datadir = '/Users/sarah/Downloads/20200923'
targname = 'Moon'
filtername = 'B'
im = fits.getdata(Bmoon[0])
image_stack = np.zeros([im.shape[0],im.shape[1],3])
#for index, fil in enumerate(Bmoon):
#    a  = fits.getdata(fil)
#    image_stack[:,:,index] = a
#
#median_image = np.nanmedian(image_stack, axis=2)
#fits.writeto(datadir + '/Stacked/' + targname + '_' + filtername + 'stack.fits', median_image, overwrite=True)
    
    


# Define the list of target(s) and the filter(s) you would like to align.
targs =['Moon']

filters=['B','V','R']  # In order of increasing wavelength: BGR, not RGB

# Provide the path to your data directory, the folder with your individual reduced science images:
# The path format should be the absolute path, e.g., "/Users/StudentName/AST337/MyData" (note there is no '/' at the end!)
datadir = '/Users/sarah/Downloads/20200923'

centerx = 2048
centery = 2048

def cross_image(im1, im2, centerx, centery, **kwargs):
    """
    Replace this with your own docstring that describes the inputs and methods used in the cross_image function. 
    Also complete any unfinished code and unfinished comments below.
    """
    
    # The type cast into 'float' is to avoid overflows:
    im1_gray = im1.astype('float')
    im2_gray = im2.astype('float')

    # Enable a trimming capability using keyword argument option.
    if 'boxsize' in kwargs:
        im1_gray = im1_gray[centery-kwargs['boxsize']:centery+kwargs['boxsize'],centerx-kwargs['boxsize']:centerx+kwargs['boxsize']]
        im2_gray = im2_gray[centery-kwargs['boxsize']:centery+kwargs['boxsize'],centerx-kwargs['boxsize']:centerx+kwargs['boxsize']]
        
    # Subtract the averages of im1_gray and im2_gray from their respective arrays -- cross-correlation
    # works better that way.
    # Complete the following two lines:
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)

    # Calculate the correlation image using fast Fourier transform (FFT)
    # Note the flipping of one of the images (the [::-1]) - this is how the convolution is done.
    corr_image = scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')
    
    # To determine the location of the peak value in the cross-correlated image, complete the line below,
    # using np.argmax on the correlation image:
    peak_corr_index = np.argmax(corr_image)

    # Find the peak signal position in the cross-correlation -- this gives the shift between the images.
    corr_tuple = np.unravel_index(peak_corr_index, corr_image.shape)
    
    # Calculate shifts (not cast to integer, but could be).
    xshift = corr_tuple[0] - corr_image.shape[0]/2.
    yshift = corr_tuple[1] - corr_image.shape[1]/2.

    return xshift,yshift



def shift_image(image,xshift,yshift):
    # Note that this will not do any trimming, 
    # so we'll want to  trim later the edges of the image using the maximum shift.
    return np.roll(np.roll(image,int(yshift),axis=1), int(xshift), axis=0)




# Cycle through list of targets:

for targname in targs:
    print(' ')
    print('-----------------------------')      
    print('target: ', targname)
    print('-----------------------------')      

    # Using glob, make list of all reduced images of current target in all filters.
    # Complete the following line to create a list of the correct images to be shifted (use wildcards!):
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
    
    print('\n') # adding some space to the print statements
    
    xshifts = {}
    yshifts = {}

    for index,filename in enumerate(imlist):
        im,hdr = fits.getdata(filename,header=True)
        xshifts[index], yshifts[index] = cross_image(im1, im, centerx, centery, boxsize=2000)
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
        
        # Complete the for loop below that ensures that each of the scilist entries has the right filter:
        for fitsfile in scilist:
            filt = fits.getheader(fitsfile)['FILTER']
            print(filt)
            
            
        
        nfiles = len(scilist)
        print('Stacking ', nfiles, filtername, ' science frames')

        # Define new array with same size as master image
        image_stack = np.zeros([im1.shape[0],im1.shape[1],len(scilist)])

        # Now that we have created an "empty" array, what is the following for loop doing?
        # Your answer: 
        
        xshifts_filt = {}
        yshifts_filt = {}
        for index,filename in enumerate(scilist):
            im,hdr = fits.getdata(filename,header=True)
            xshifts_filt[index], yshifts_filt[index] = cross_image(im1, im, centerx, centery, boxsize=800)
            image_stack[:,:,index] = shift_image(im,xshifts_filt[index], yshifts_filt[index])

        # Complete the line below to take the median of the image stack (median combine the stacked images);
        # Be careful to use the correct 'axis' keyword in the np.median function!
        median_image = np.median(image_stack,axis=2)

        # Sets the new image boundaries
        if (max_x_shift > 0) & (max_y_shift > 0): # don't apply cut if no shift!
            median_image = median_image[max_x_shift:-max_x_shift,max_y_shift:-max_y_shift]

        # Make a new directory in your datadir for the new stacked fits files
        if os.path.isdir(datadir + '/Stacked') == False:
            os.mkdir(datadir + '/Stacked')
            print('\n Making new subdirectory for stacked images:', datadir + '/Stacked \n')
            
        
        # Save the final stacked images into your new folder:
        fits.writeto(datadir + '/Stacked/' + targname + '_' + filtername + 'stack.fits', median_image, overwrite=True)
        print('   Wrote FITS file ',targname+'_'+filtername+'stack.fits', 'in ',datadir + '/Stacked/','\n')
        
print('\n Done stacking!')

            
            