# ACO pipeline part 3
# ACO photometry pipeline 

# written by Sarah Betti 2020
# bg_error_estimate, starExtractor, measurePhotometry written by Kim Ward-Duong 2018-2019

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from astropy.io import fits 
import astropy.stats as stat
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval
from astropy.stats import mad_std

from photutils.utils import calc_total_error
from photutils import aperture_photometry, CircularAperture, CircularAnnulus, DAOStarFinder

import glob



def bg_error_estimate(fitsfile):
    """
    This function will calculate the noise in the background of our images 
    """
    fitsdata = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)
    
    # We are finding the amount of noise in our image
    filtered_data = sigma_clip(fitsdata, sigma=3.,copy=False)
    
    # We are calculating the median value of the noise
    bkg_values_nan = filtered_data.filled(fill_value=np.nan)
    bkg_error = np.sqrt(bkg_values_nan)
    bkg_error[np.isnan(bkg_error)] = np.nanmedian(bkg_error)
    
    effective_gain = 1.4 # electrons per ADU
    
    error_image = calc_total_error(fitsdata, bkg_error, effective_gain)  
    
    return error_image

def starExtractor(fitsfile, nsigma_value, fwhm_value):
    """
    This function find all the stars in our image and gives the x and y position of the star. 
    """

    # Read in the data from the fits file 
    image = fits.getdata(fitsfile)
    
    # Measure the median absolute standard deviation of the image: We did it above, but copy and paste it below 
    bkg_sigma = mad_std(image)

    # Define the parameters for DAOStarFinder 
    daofind = DAOStarFinder(fwhm=fwhm_value, threshold=nsigma_value*bkg_sigma)
    
    # Apply DAOStarFinder to the image
    sources = daofind(image)
    nstars = len(sources)
    print("Number of stars found in ",fitsfile,":", nstars)
    
    # Define arrays of x-position and y-position
    xpos = np.array(sources['xcentroid'])
    ypos = np.array(sources['ycentroid'])
    
    return xpos, ypos # Return the x and y positions of each star as variables

def measurePhotometry(fitsfile, star_xpos, star_ypos, aperture_radius, sky_inner, sky_outer, error_array):
    """
    find the amount of flux from each star. 
    """
    # Read in the data from the fits file:
    image = fits.getdata(fitsfile)
    
    pos = [(star_xpos[i],star_ypos[i]) for i in np.arange(len(star_xpos))]
    starapertures = CircularAperture(pos,r = aperture_radius)
    skyannuli = CircularAnnulus(pos, r_in = sky_inner, r_out = sky_outer)
    phot_apers = [starapertures, skyannuli]
    
    phot_table = aperture_photometry(image, phot_apers, error=error_array)
        
    # Calculate mean background in annulus and subtract from aperture flux
    bkg_mean = phot_table['aperture_sum_1'] / skyannuli.area
    bkg_starap_sum = bkg_mean * starapertures.area
    final_sum = phot_table['aperture_sum_0']-bkg_starap_sum
    phot_table['bg_subtracted_star_counts'] = final_sum
    
    bkg_mean_err = phot_table['aperture_sum_err_1'] / skyannuli.area
    bkg_sum_err = bkg_mean_err * starapertures.area

    phot_table['bg_sub_star_cts_err'] = np.sqrt((phot_table['aperture_sum_err_0']**2)+(bkg_sum_err**2)) 
    
    return phot_table

def extract_photometry(im, X, Y):
    '''
    extract photometry from star positions
    '''

    # Measure the background of the image
    std_F_bgerror = bg_error_estimate(im)

    # Measure photometry for each band image.
    std_F_phottable = measurePhotometry(im, star_xpos=X, 
                                        star_ypos=Y, aperture_radius=8, sky_inner=10, sky_outer=15, error_array=std_F_bgerror)
    return std_F_phottable

def instr_mag(fluxtable, exptime, filt):
    '''
    calculate instrumental flux and add to fluxtable 
    '''
    fluxtable[filt+'flux_1sec'] = fluxtable[filt+'flux'] / exptime

    fluxtable[filt+'flux_1sec_err'] = fluxtable[filt+'flux_1sec']*(fluxtable[filt+'fluxerr']/fluxtable[filt+'flux'])

    fluxtable[filt+'_inst'] = -2.5*np.log10(fluxtable[filt+'flux_1sec'])
    fluxtable[filt+'inst_err'] = -2.5*0.434*(fluxtable[filt+'flux_1sec_err']/fluxtable[filt+'flux_1sec'])
    
    return

########################## 

def run_photometry(fil, clustername, standardname, plot=True):
    '''
    run photometry for cluster and standard star
    '''
    
    print()
    print('--------o----------')
    print('starting photometry')
    print()
    
    # grab stacked frames
    cluster = np.sort(glob.glob(fil + '/Stacked/' + clustername + '*.fits'))
    standard = np.sort(glob.glob(fil + '/Stacked/' + standardname + '*.fits'))
    
    # calculate x and y positions
    std_F1_xpos, std_F1_ypos = starExtractor(standard[2], nsigma_value=10, fwhm_value=10)
    
    cluster_F1_xpos, cluster_F1_ypos = starExtractor(cluster[2], nsigma_value=10, fwhm_value=10)

    if plot:
        #Plot your standard star and photometry.
        pos_std = [(std_F1_xpos[i],std_F1_ypos[i]) for i in np.arange(len(std_F1_xpos))]
        pos_clu = [(cluster_F1_xpos[i],cluster_F1_ypos[i]) for i in np.arange(len(cluster_F1_xpos))]
        apertures_std = CircularAperture(pos_std, r=30)
        apertures_clu = CircularAperture(pos_clu, r=30)
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)
        axes = [ax1, ax2]
        im = [apertures_clu,apertures_std]
        star=[cluster, standard]
        titles = ['cluster', 'standard star']
        for i in range(2):
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(fits.getdata(star[i][2]))
            axes[i].imshow(fits.getdata(star[i][2]), vmin=vmin,vmax=vmax, origin='lower')
            im[i].plot(color='white', lw=2, axes=axes[i])
            axes[i].set_title(titles[i] + ' image with apertures')
        plt.show()

    # go through each frame and extract the photometry and put into a fluxtable
    for i in np.arange(len(cluster)):
        cluster_F_phottable = extract_photometry(cluster[i], cluster_F1_xpos, cluster_F1_ypos)
        
        std_F_phottable = extract_photometry(standard[i], std_F1_xpos, std_F1_ypos)
        
        filt_cluster = fits.getheader(cluster[i])['FILTER'][0]
        filt_standard = fits.getheader(standard[i])['FILTER'][0]
        if i == 0:
            cluster_fluxtable = pd.DataFrame(
                {'id'      : cluster_F_phottable['id'],
                 'xcenter' : cluster_F_phottable['xcenter'],
                 'ycenter' : cluster_F_phottable['ycenter']})
            
            standard_fluxtable = pd.DataFrame(
                {'id'      : std_F_phottable['id'],
                 'xcenter' : std_F_phottable['xcenter'],
                 'ycenter' : std_F_phottable['ycenter']})
            
        cluster_fluxtable[filt_cluster + 'flux'] = cluster_F_phottable['bg_subtracted_star_counts']
        cluster_fluxtable[filt_cluster + 'fluxerr']=  cluster_F_phottable['bg_sub_star_cts_err'] 
            
        standard_fluxtable[filt_standard + 'flux'] = std_F_phottable['bg_subtracted_star_counts']
        standard_fluxtable[filt_standard + 'fluxerr']=  std_F_phottable['bg_sub_star_cts_err'] 
        
    # save as csv 
    standard_fluxtable.to_csv(fil + '/Stacked/' + standardname + '_photometry.csv')
    cluster_fluxtable.to_csv(fil + '/Stacked/' + clustername + '_photometry.csv')

def calibrate_photometry(fil,clustername, standardname, **kwargs):
    '''
    calibrate photometry using real filter magnitudes
    '''
    # open files and frames
    standard_fluxtable = pd.read_csv(fil + '/Stacked/' + standardname + '_photometry.csv')
    cluster_fluxtable=  pd.read_csv(fil + '/Stacked/' + clustername + '_photometry.csv')

    cluster = np.sort(glob.glob(fil + '/Stacked/' + clustername + '*.fits'))
    standard = np.sort(glob.glob(fil + '/Stacked/' + standardname + '*.fits'))
  
    # find standard star in frame 
    if kwargs.get('plot'):
        plt.figure()
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(fits.getdata(standard[0]))
        plt.imshow(fits.getdata(standard[0]), origin='lower', vmin=vmin, vmax=vmax)
        plt.show()
        
    # find standard star x and y positon in fluxtable 
    if 'standard_x_y' in kwargs:
        pos = kwargs['standard_x_y']
    else:
        pos = input('x,y position of standard star.  Separate by space: ')
    x = int(pos.split(' ')[0])
    y = int(pos.split(' ')[1])

    print( standard_fluxtable.loc[(standard_fluxtable['xcenter']>x-5) & (standard_fluxtable['xcenter']<x+5) &(standard_fluxtable['ycenter']>y-5) & (standard_fluxtable['ycenter']<y+5)]  )
    
    if 'standard_row' in kwargs:
        row = kwargs['standard_row']
    else:
        row = int(input('row of standard star: '))
    XX = standard_fluxtable['xcenter'][row]
    YY = standard_fluxtable['ycenter'][row]
    
    if kwargs.get('plot'):
        # replot your standard star image
        plt.figure()
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(fits.getdata(standard[0]))
        plt.imshow(fits.getdata(standard[0]), origin='lower', vmin=vmin, vmax=vmax)
        circle = plt.Circle((XX, YY), 30, color='red', fill=None)
        plt.gca().add_artist(circle)
        plt.show()


    std_star_row = standard_fluxtable.loc[row]
    # save to new dataframe
    std_star_fluxtable = pd.DataFrame({'standard star':std_star_row}).T
    
    
    # calculate instrumental magnitude
    star=[standard, cluster]
    table = [std_star_fluxtable,cluster_fluxtable]

    for i in np.arange(len(standard)):
        exptime = fits.getheader(standard[i])['EXPTIME']
        filt = fits.getheader(standard[i])['FILTER'][0]
        instr_mag(std_star_fluxtable, exptime, filt)
        
    print(std_star_fluxtable)
    # use real magnitudes to calibrate cluster
    for i in np.arange(len(cluster)):    
        filt = fits.getheader(cluster[i])['FILTER'][0]
        exptime = fits.getheader(cluster[i])['EXPTIME']
        instr_mag(cluster_fluxtable, exptime, filt)
        
            
        if filt+'_mag' in kwargs:
            mags = kwargs[filt+ '_mag']
        else:
            mags=input('real std magnitude of filter ' + filt + ': ')
        mag = np.float(mags.split(' ')[0])
        mag_err = np.float((mags.split(' ')[1]).split('[')[1].split(']')[0])
        inst_mag = std_star_fluxtable[filt+'_inst'].values[0]
        inst_mag_err = std_star_fluxtable[filt+'inst_err'].values[0]
        print(filt, exptime, inst_mag, inst_mag_err)

        magzp = mag - inst_mag
        magzp_error = np.sqrt(mag_err**2. + (inst_mag_err)**2.)
        print(f"Zeropoint in {filt}: ", magzp, "+/-", magzp_error)
        
        cluster_fluxtable[filt+'_mcal'] = cluster_fluxtable[filt+'_inst'] + magzp

        cluster_fluxtable[filt+'_mcal_err'] = np.sqrt((cluster_fluxtable[filt+'inst_err'])**2 + magzp_error**2)
        
    # save to csv file
    cluster_fluxtable.to_csv(fil + '/Stacked/' + clustername + '_photometry_final.csv')    
            
def plot_CMD(fil, clustername, filt1, filt2, save=False):
    '''
    plot CMD for 2 filters
    '''
    cluster = pd.read_csv(fil + '/Stacked/' + clustername + '_photometry_final.csv')

    cluster[filt1+'-'+filt2] = cluster[filt1+'_mcal'] - cluster[filt2+'_mcal']

    cluster[filt1 + '-' +filt2+ '_err'] = np.sqrt((cluster[filt1+'_mcal_err'])**2 + (cluster[filt2+'_mcal_err'])**2)

    X = cluster[filt1+'-'+filt2]
    Xerr = cluster[filt1 + '-' +filt2+ '_err']

    Y = cluster[filt2+ '_mcal']
    Yerr = cluster[filt2 + '_mcal_err']

    plt.errorbar(X, Y, xerr=Xerr, yerr=Yerr, marker='.', color='k',linestyle='None', alpha=0.5, ecolor='gray')
    plt.gca().invert_yaxis()
    plt.title(clustername, fontsize=16)
    plt.xlabel(filt1 + '-' + filt2 + ' [mag]', fontsize=16)
    plt.ylabel(filt2 + ' [mag]', fontsize=16)
    plt.tick_params(which='both', direction='in', labelsize=16, top=True, right=True)
    plt.minorticks_on()
    plt.xlim(-0.8, 3)
    plt.ylim(18, 6)
    


    if save:
         plt.savefig(f'{fil}/Stacked/{clustername}_CMD.pdf')
    plt.show()















