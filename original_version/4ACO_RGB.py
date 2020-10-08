
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy import stats
from PIL import Image

# EDIT AS NEEDED

targname = 'M13'
filters = ['B','V','R']  # In order of increasing wavelength: BGR, not RGB

# Provide the path to your data directory, the folder with your *stacked* reduced science images:
# The path format should be the absolute path, e.g., "/Users/StudentName/AST337/MyData/Stacked/" 

# (note there IS a '/' at the end this time!)
datadir = '/Users/sarah/Downloads/20200911/Stacked/'

siglowhi = [-1,20.,-1,20.,-1,20.] # BGR low and high sigma limits. Ex: To make red brighter, make 6th # lower.

Rtmp = fits.getdata(datadir+targname+'_'+filters[2]+'stack.fits')
print(np.mean(Rtmp))
print(datadir+targname+'_'+filters[2]+'stack.fits')


def scale_filter(tmpimg,lowsig=-1.,highsig=15.):

    tmpimg -= np.median(tmpimg)
    print('minmax 1: ', np.min(tmpimg),np.max(tmpimg))


    tmpsig = stats.sigma_clipped_stats(tmpimg, sigma=2, maxiters=5)[2]
    print('std: ', tmpsig)
    print("lowsig, highsig: ", lowsig, highsig)
    print('cuts: ', lowsig*tmpsig, highsig*tmpsig)

    image_hist = plt.hist(tmpimg.flatten(), 1000, range=[-100,100])

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

# Read in 3 images 
Rtmp = fits.getdata(datadir+targname+'_'+filters[2]+'stack.fits')
Gtmp = fits.getdata(datadir+targname+'_'+filters[1]+'stack.fits')
Btmp = fits.getdata(datadir+targname+'_'+filters[0]+'stack.fits')

# Scale all 3 images
print('Calculating stats....')
R = scale_filter(Rtmp,lowsig=siglowhi[4],highsig=siglowhi[5])
G = scale_filter(Gtmp,lowsig=siglowhi[2],highsig=siglowhi[3])
B = scale_filter(Btmp,lowsig=siglowhi[0],highsig=siglowhi[1])

# Merge 3 images into one RGB image
im = Image.merge("RGB", (R,G,B))

im.save(datadir+targname+'_RGB.jpg', "JPEG")
print("Saved image as ", datadir+targname+'_RGB.jpg')