# -*- coding: utf-8 -*-
"""
fits all the resonances of a spectrum
You will ned the new scipy to use the fit using boundaries
I use anaconda with python3 64 bits

"""

from pylab import *
from scipy.io import loadmat
import pylab as pylab
import kasey_fitspectra as kcfit
import numpy
#import pandas as pd

# for autocorr:
from matplotlib.mlab import find
from scipy.signal import fftconvolve
from parabolic import parabolic

from scipy.optimize import curve_fit
#from scipy.optimize import least_squares
#from optimize import curve_fit

import matplotlib.pyplot as plt

#PATH = 'C:\\data\\2016_edx_25C\\data\\'
#PATH = 'C:\\data\\2015b_edx_25C\\'
#FPATH = 'C:\\Users\\matres\\Dropbox\\papers\\backscattering\\figures\\'


PATH = '/Users/joaquin/photonics/data/2015b_edx_25/'
FPATH = '/Users/joaquin/Dropbox/papers/backscattering/figures/'


L_rt = 2*pi*5
PDF = True


def f_ring(wav, K = 1, RTL = 0.1, wav0 = 1550):
    """
    K1: coupling power coefficient
    
    
    wav in nm
    Wayne nomenclature
    R1 is the reflection coefficient of the first coupler (through) in Power
    T1 is the transmitted power into the cavity (coupler coefficient)
    (wav, R1 = 0.99, R2 = 0.99, Ta = 0.999, phi0 = pi+ 0.2, bend_radius = 5, neff = 2.4, logea = True)
    """

    bend_radius = 5
    neff = 2.4
    logea = True

    R1 = (100-K)*1e-2
    R2 = (100-K)*1e-2
    T1 = 1-R1
    T2 = 1-R2
    Tp = (100-RTL)*1e-2
    
#    L_rt =  L#2*pi*bend_radius   # Lc*2                    # round trip length
    phi  = (2*pi/(wav*1e-3))*neff*L_rt - (2*pi/(wav0*1e-3))*neff*L_rt

    T1 = 1-R1
    Tc = R1*R2*Tp**2

    tp1 = sqrt(Tp)
    t1 = sqrt(T1)
    t2 = sqrt(T2)
    r1 = sqrt(R1) 
    tc = sqrt(Tc)*exp(1j*phi)

    Hr = abs(sqrt(R1) - T1*sqrt(Tc)*exp(-1j*phi)/sqrt(R1)/(1-sqrt(Tc)*exp(-1j*phi)))**2
    Hc = 1j*t1/(1-tc)
    Ht = r1 + 1j*t1*tc/r1*Hc
    
    if logea:
        Hr = 10*log10(abs(Hr))

#    plot(wav, Hr)
    return Hr


def f_ring_bs(wav, K = 1, RTL = 0.01, Rbs = 20, wav0 = 1550):
    """
    changed -Rbs to dB    
    includes backscattering model
    K is the coupler coupling coefficient in (%)
    % it is an add drop ring, so we assume K1 = K2 =K
    
    """
    bend_radius = 5
    neff = 2.4
    logea = True
    
    R1 = (100-K)*1e-2
    R2 = (100-K)*1e-2
    T1 = 1-R1
    T2 = 1-R2
    Tp = (100-RTL)*1e-2
    
#    L_rt =  2*pi*bend_radius   # Lc*2                    # round trip length
    phi  = (2*pi/(wav*1e-3))*neff*L_rt - (2*pi/(wav0*1e-3))*neff*L_rt # phi = beta*L

    tp = sqrt(Tp)
    t1 = sqrt(T1)
    t2 = sqrt(T2)
    r1 = sqrt(R1)

    rbs = 10**(-Rbs/20.0) # L_rt*1e-6*1e3    
    tbs = sqrt(1-rbs**2)
    Tbs = tbs**2
    
    Tc = R1*R2*Tp*Tbs
    tc = sqrt(Tc)*exp(1j*phi)
    
    Hc = 1j*t1/(1-tc)/(1+tc*(rbs/tbs/(1-tc))**2)
    Ht = r1 + 1j*t1*tc/r1*Hc
    Hd = 1j*t2*rbs*tp*Hc

    if logea:
        Ht = 20*log10(abs(Ht))
    
    return Ht


class Spectrum():
    """
    This class loads in a .mat and stores the wavelength and power respectively.
    wavelen: contains the wavelenghts in nm
    lum: power in dBm

    Various methods allow for plotting the data [*self.plot()*], normalizing it [*self.normalize()*], etc.
    read: reads the spectrum into (wavelen, lum), saves an original copy into (w,p)
    plot: plots the spectrum (wavelen, lum)
    chop(wav min, wav max): chops the spectrum lum
    fit_gaussians: 
    fit_lorentzians:
    get_fitrange_indices:
    normalize:
    fit_fsr: Estimates fsr using autocorrelation
    peakdet: detects the dips, plotea = True plots dips, plot_FSR plots FSR
    fit_resonances:
    
    Based on http://people.seas.harvard.edu/~krussell/html-tutorial/plotscript1-3.html
    http://people.seas.harvard.edu/~krussell/html-tutorial/_modules/winspec.html#Spectrum
    https://github.com/kaseyrussell/python_misc_modules    
    """
    def __init__(self, fname, index = 0, path = PATH, remove_baseline = True):
        self.fname          = fname
        self.ax             = None
        self.fit_params     = None
        self.laser          = None
        self._slope_remove  = False
        self._slope_npoints = None
        self.remove_baseline   = remove_baseline
        self.baselined  = False
        self.read(fname, index, path)

        
    def read(self, fname, index = 0, path = PATH):
        """ 
        Read the data
        Saves power and wavelength in self.power and self.wavelen.
        Saves SPE file header info in self.header        
        If you pass it the additional argument
        dB=True
        then it will return the spectrum in counts per second rather than
        just counts (the default).
        """
        
        matData = loadmat(path + fname, squeeze_me=True, struct_as_record=False) 
        self.lum =  matData['scandata'].power[:,index]
        self.wavelen =  matData['scandata'].wavelength*1e9
        
        if self.remove_baseline:
            self.lum = self.baseline()

        self.p = array(self.lum)# original copies of the data   
        self.w = array(self.wavelen)


    def reset_spectrum(self):
        self.lum = array(self.p)
        self.wavelen = array(self.w)


    def baseline(self, N = 5):
        """
        for an already-loaded spectrum, remove the grating coupler spectrum
        contained in the file fname.
        N is the order of the polynomial fit
        """
        self.baselined = True
        w0 = 1550
        w = self.wavelen
        p = self.lum
        
        c = polyfit(w-w0, p, N)
        p_fit = polyval(c, w-w0)
        
        self.lum -= p_fit
        return self.lum
        

    def chop( self, start=None, stop=None ):
        """
        restrict the spectrum to only this wavelength range
        (start and stop are in nm)
        """
        if start is None:
            start = self.wavelen[0]
        
        if stop is None:
            stop = self.wavelen[-1]
        
        if start < self.wavelen[0] or stop > self.wavelen[-1]:
            raise ValueError("wavelength range is not contained within\
                                the spectrum.")
        
        self.lum = self.lum[ pylab.find(self.wavelen > start) ]
        self.wavelen = self.wavelen[ find(self.wavelen > start) ]
        
        self.lum = self.lum[ pylab.find(self.wavelen < stop) ]
        self.wavelen = self.wavelen[ find(self.wavelen < stop) ]
        
        self.lum -= max(self.lum)
#        return self.lum
#        return [self.wavelen, self.lum]
        return [array(self.wavelen), array(self.lum)] # returns a copy of the chopped data


    def fit_gaussians( self, plotfit=True, halfwidth=5.0, width=None,
                       center=None, height=None, yoffset=0.0, printparams=True ):
        """
        By default (i.e. unless 'center' is specified), this fits Gaussians
        using the function *easyfitgaussians* from the module
        *kasey_fitspectra*. This will then ask you to click on the peaks you
        would like to fit.
        
        The parameter 'halfwidth' specifies the initial guess at the peak
        halfwidth (in nm). 
        
        Alternatively, you can pass the keyword argument 'width'
        (equal to twice the halfwidth). If both are specified, the
        value for 'width' will be used.

        The fit parameters are stored in self.fitparams as a
        list of dictionaries, one for each peak, with the following
        keys:
            
            'yoffset', 'ymax', 'halfwidth', 'x0'

        where yoffset and ymax are the vertical/lum. offset and
        the lum. peak height, respectively,
        and x0 is the peak wavelength
        (by halfwidth, I mean standard deviation.)

        By default (plotfit=True), it plots the result as a black line
        of width 2. To override this, pass plotfit=False.

        If 'center' is specified, then the fit will not be interactive, and
        the data will be fit with a single lorentzian centered at 'center' and
        of amplitude 'height' (which, if not specified, will be taken as the
        value of the data at the point 'center').

        By default (printparams=True), the fit parameters are printed to screen.
        """

        if center is None:
            withmouse = True
        else:
            withmouse = False
            if center < self.wavelen.min():
                center = self.wavelen.min()
            if center > self.wavelen.max():
                center = self.wavelen.max()
            
        if width is not None:
            halfwidth = width/2
            
        if withmouse==True:
            result = kcfit.easyfitgaussians( self.wavelen, self.lum, 
                                               halfwidth )
            self.bestfit, self.fit_params = result
            
        else:
            #For a single Gaussian, params can be a dict containing
            #the keys yoffset, ymax, x0, and halfwidth.
            if height is None:
                dw = self.wavelen[1] - self.wavelen[0]
                height = self.lum[ pylab.where( abs(self.wavelen-center)<dw/2 ) ]
            
            initparams = dict( yoffset=yoffset,
                               ymax=height,
                               halfwidth=halfwidth,
                               x0=center )

            result = kcfit.fitgaussians( self.wavelen, self.lum, initparams )
            self.bestfit, pbest, std_err = result
            
            kcfit.absparams(pbest)
            self.fit_params = kcfit.params_to_dicts( kcfit.splitparams(pbest), 
                                                    kcfit.splitparams(std_err) )
            
        if plotfit==True:
            self.plot_fit( linewidth=2.0, color='black' )

        if printparams==True:
            self.print_fit_parameters()
    
    

    def fit_lorentzians( self, plotfit=True, halfwidth=0.1, width=None,
                       center=None, height=None, yoffset=0.0, printparams=True,
                       fitrange=(0,-1), number_bestfit_points=None ):
        """
        Fit one or more Lorentzian functions to the data. If you use the *center* parameter,
        the fit will not be interactive and will attempt to fit based on the info passed in.
        If you do not use the *center* parameter, then the function *easyfitlorentzians* from
        the module *kasey_fitspectra* will be used, and it will ask you to click on peaks that
        you would like to fit. That case does not allow you to restrict the fitting range, though,
        so the entire spectrum must be fit, which is often not what you want.
        
        *plotfit*
            Boolean value (default=True) plots the resulting fit to the current axes.

        *halfwidth*
            Guess at peak halfwidth in nm (default=5.0). For multiple peaks,
            a single value can be used for all peaks or a list of values can
            be specified, one for each peak.
            Alternatively, you can pass the keyword argument
            *width* (equal to twice the halfwidth). If both are specified, the
            value for 'width' will be used.
                    
        *width*
            Guess at peak width in nm. If both *width* and *halfwidth* are specified,
            *width* will be used.

        *center*
            Guess at center point in nm for the peak. For multiple peaks, a
            list of values should be supplied.
            
        *height*
            Guess at height of peak. If not supplied, will be taken as the
            value of the data at the point *center*.

        *yoffset*
            Guess at y-axis offset of the base of the Lorentzian. Default is 0.
            
        *printparams*
            Boolean value (default=True) specifying whether the resulting fit parameters should
            be printed to the terminal.

        *fitrange*
            Two-element tuple (*start, stop*) containing the array indices specifying
            the beginning and end of the fit range. Default is (0,-1), the full range.
        
        *number_bestfit_points*
            The number of points to use in the best fit line. Default is to use the same
            number as the data. You might want more if you have a really sharp peak and
            want to have a smooth Lorentzian curve over it, for example.

        The fit parameters are stored in self.fitparams as a
        list of dictionaries, one for each peak, with the following
        keys:
            *yoffset*, *ymax*, *halfwidth*, *x0*, *Q*
        where *yoffset* and *ymax* are the vertical/lum. offset and
        the lum. peak height, respectively,
        and *x0* is the peak wavelength

        """

        if center is None:
            withmouse = True
            if width is not None: halfwidth = width/2
        else:
            withmouse = False
            if len(numpy.shape(center)) == 0: center = [center]
            if len(numpy.shape(halfwidth)) == 0: halfwidth = [halfwidth]*len(center)
            if len(numpy.shape(yoffset)) == 0: yoffset = [yoffset]*len(center)
            if width is not None:
                if len(numpy.shape(width)) == 0:
                    halfwidth = [width/2]*len(center)
                else:
                    halfwidth = [item/2 for item in width]
                
            for i,item in enumerate(center):
                if item < self.wavelen.min(): center[i] = self.wavelen.min()
                if item > self.wavelen.max(): center[i] = self.wavelen.max()
            
            
        if withmouse==True:
            result = kcfit.easyfitlorentzians( self.wavelen, self.lum, 
                                               halfwidth )
            self.bestfit, self.fit_params = result
            
        else:
            """
            The initial paramters is a list of dicts, with each dict
            containing the parameters for a single lorentzian with
            the keys yoffset, ymax, x0, and halfwidth.
            """ 

            if height is None:
                dw = self.wavelen[1] - self.wavelen[0]
                height = [self.lum[ pylab.where( abs(self.wavelen-x0)<dw/2 ) ] for x0 in center]
            elif len(numpy.shape(height))==0:
                height = [height]
            
            initparams = []
            for y0, ymax, dx, x0 in zip(yoffset, height, halfwidth, center):
                initparams.append(dict( yoffset=y0,
                                   ymax=ymax,
                                   halfwidth=dx,
                                   x0=x0 ))

            """ 'values' is the array of y-axis values """
            values = self.lum[fitrange[0]:fitrange[1]].copy()


            """ optionally remove a linear bg before fitting lorentzians """
            if self._slope_remove:
                npoints = self._slope_npoints
                subset_x = self.wavelen[fitrange[0]:fitrange[1]]
                subset_y = self.lum[fitrange[0]:fitrange[1]]
                if npoints is None:
                    linearfit = numpy.polyfit( 
                        self.wavelen[fitrange[0]:fitrange[1]], 
                        self.lum[fitrange[0]:fitrange[1]], 
                        deg=1 )
                else:
                    x = numpy.zeros( 2*npoints )
                    x[:npoints] = subset_x[:npoints]
                    x[npoints:] = subset_x[-npoints:]
                    
                    y = numpy.zeros( 2*npoints )
                    y[:npoints] = subset_y[:npoints]
                    y[npoints:] = subset_y[-npoints:]
                    
                    linearfit = numpy.polyfit( x, y, 1 )

                linear_bg = numpy.polyval( linearfit, self.wavelen[fitrange[0]:fitrange[1]] )
                values -= linear_bg


            """ Do the actual Lorentzian fit. """
            bestfit, self.pbest, std_err = kcfit.fitlorentzians(
                self.wavelen[fitrange[0]:fitrange[1]],
                values, 
                initparams )
            
            """ Optionally increase the resolution of the bestfit points """
            if number_bestfit_points is None:
                self.bestfit = bestfit
                self.fit_wavelen = self.wavelen
            else:
                self.fit_wavelen = numpy.linspace(
                    self.wavelen[fitrange[0]],
                    self.wavelen[fitrange[1]],
                    number_bestfit_points)
                self.bestfit = kcfit.lorentzians(self.fit_wavelen, *self.pbest)


            """ Make all parameters > 0 for easy ordering (those squared within the function
            can come back as +/-). I abs() all of them out of lasiness, but this would be bad if we
            had a negative yoffset or a dip rather than a peak. You don't really need to do this, 
            though, so if it causes problems just comment it out."""
            kcfit.absparams(self.pbest)
            self.fit_params = kcfit.params_to_dicts( kcfit.splitparams(self.pbest), 
                                                    kcfit.splitparams(std_err) )
            
            for peak in self.fit_params:
                peak['Q'] = peak['x0']/2/peak['halfwidth']
                # error in Q is returned as a min/max tuple of the standard error
                peak['Q_err'] = numpy.asarray( (peak['x0']-peak['x0_err'])/2/(peak['halfwidth']+peak['halfwidth_err']),
                                         (peak['x0']+peak['x0_err'])/2/(peak['halfwidth']-peak['halfwidth_err']) )
                peak['Q_err'] = numpy.abs( peak['Q'] - peak['Q_err'] )
        
                if self._slope_remove:
                    peak['slope_removed'] = True
                    peak['slope_npoints'] = self._slope_npoints
                    peak['slope_fit'] = linearfit
                

        if plotfit:
            self.plot_fit( linewidth=2.0, color='black', fitrange=fitrange )

        if printparams:
            self.print_fit_parameters()
    

    def get_fitrange_indices( self, fitrange_nm ):
        """
        Convert a wavelength range (*start*, *stop*) into a range of array indices.
        This algorithm doesn't check that your input is valid and doesn't try to
        figure out which point you're closest to, only the largest one you are larger than.
        """
        start, stop = fitrange_nm
        if start >= self.wavelen[-1]:
            raise ValueError ("Start value too high.")
        start_index = pylab.where(self.wavelen > start)[0][0]

        if stop < self.wavelen[-1]:
            stop_index = pylab.where(self.wavelen > stop)[0][0]
        else:
            stop_index = -1

        return start_index, stop_index


    def normalize( self, wavelen=None, value=None ):
        """ Multiplies the spectrum by 'value'.
            
            If value is None, value is set to 1/lum.max() to
            normalize the intensity to unity.
            
            If wavelen=None (default), this just divides the
            luminescence intensity by its peak value.
            
            If wavelen is specified, then the intensity is
            normalized to the intensity at that wavelength
            rather than at the peak.

            To specify an endpoint of the spectrum, simply
            set wavelen outside the wavelength range:
            if wavelen is less than self.wavelen.min(), then
            self.wavelen.min() is used (and ditto for max value).
            
            The value multiplied to normalize the spectrum will
            be stored in self.scale_factor, e.g. 1/lum.max() by default.
            
        """
        if wavelen is None and value is None:
            self.scale_factor = 1.0/self.lum.max()
            self.lum /= self.lum.max()
        elif wavelen is not None:
            if wavelen < self.wavelen.min():
                self.lum /= self.lum[0]
                self.scale_factor = 1.0/self.lum[0]
            elif wavelen > self.wavelen.max():
                self.lum /= self.lum[-1]
                self.scale_factor = 1.0/self.lum[-1]
            else:
                lum_i = find( self.wavelen < wavelen )[-1]
                self.lum /= self.lum[ lum_i ]
                self.scale_factor = 1.0/self.lum[ lum_i ]
        elif value is not None:
            self.lum *= value
            self.scale_factor = value
        

    def on_close_axes( self, event ):
        """ Here we reset the self.axes flag to None so that if
        you run the *plot* method again you will get a new window
        rather than plot into a closed plot window... """
        #event.canvas.figure.axes[0].has_been_closed = True
        self.ax = None


    def plot( self, *args, **kwargs):
        """
        Plots the spectrum.
        Any extra arguments passed to this method get passed
        on to plot, making it possible to do stuff like
        self.plot( '--b', linewidth=3 )
        You can add a vertical offset to the data by setting
        yoffset = xxx
        or a horizontal offset with
        xoffset = xxx
        (e.g. to enable making a waterfall plot of many traces)
        
        if you pass fill=True, then the function 'fill_between' will
        be used instead of 'plot', and the value of 'yoffset' (if set)
        will be used as the 'y2' value for 'fill_between'
        
        if you pass baseline=True then it will be plotted as a baseline normalized spectrum
        """
        if 'yoffset' in kwargs.keys():
            voffset = kwargs['yoffset']
            del kwargs['yoffset']
        else:
            voffset = 0.0
            
        if 'xoffset' in kwargs.keys():
            hoffset = kwargs['xoffset']
            del kwargs['xoffset']
        else:
            hoffset = 0.0
            
        if 'fill' in kwargs.keys():
            fill = kwargs['fill']
            del kwargs['fill']
        else:
            fill = False
        
        if 'semilogy' in kwargs.keys():
            semilogy = kwargs['semilogy']
            del kwargs['semilogy']
        else:
            semilogy = False

        if 'connect_on_close' in kwargs.keys():
            connect_on_close = kwargs['connect_on_close']
            del kwargs['connect_on_close']
        else:
            connect_on_close = True

        x_axis = self.wavelen + hoffset
                
        if self.ax is None:
            self.ax = pylab.gca()
            
        if fill:
            line = self.ax.fill_between( x_axis, self.lum + voffset,
                         y2 = voffset, **kwargs )
        else:
            line, = self.ax.plot( x_axis, self.lum + voffset,
                         *args, **kwargs )

        if semilogy:
            self.ax.set_yscale('log')

        self.ax.set_xlabel('Wavelength (nm)')
            
        if connect_on_close:
            pylab.gcf().canvas.mpl_connect('close_event', self.on_close_axes)
        return line


    def plot_fit( self, *args, **kwargs ):
        """
        If you fitted one (or more) Lorentzians or Gaussians to your spectrum,
        this method will plot them. As with the plot method, any
        extra args passed to this method get passed directly to
        pylab's plot command.
        """
        try:
            self.bestfit
        except (AttributeError, NameError):
            print ("no fit found.")
            return
        
        if 'yoffset' in kwargs.keys():
            voffset = kwargs['yoffset']
            del kwargs['yoffset']
        else:
            voffset = 0.0
            
        if 'xoffset' in kwargs.keys():
            hoffset = kwargs['xoffset']
            del kwargs['xoffset']
        else:
            hoffset = 0.0
            
        if 'fitrange' in kwargs.keys():
            fitrange = kwargs['fitrange']
            del kwargs['fitrange']
        else:
            fitrange = (0,-1)

        if self._slope_remove:
            """ restore the linear background """
            linear_bg = numpy.polyval( self.fit_params[0]['slope_fit'], self.fit_wavelen )
            voffset += linear_bg
        
        line, = pylab.plot( (self.fit_wavelen+hoffset)[:-1], self.bestfit+voffset, *args, **kwargs )
        return line
        

    def print_fit_parameters( self ):
        """ print the fit parameters to the screen in a nice format
        """
        if self.fit_params is None:
            print ("There are no fit parameters to display.")
            return
        
        peak_number = 1
        for peak in self.fit_params:
            f0 = peak['x0']
            Q = peak['Q']
            print(peak['ymax'])
            R0 = 10**(-peak['ymax']/10.0)
            mc = f0/self.fsr
            loss_c = 2*pi*mc/Q # cavity loss
            loss_1 = (1-sqrt(R0))*loss_c/2.0 #sqrt(R0) = (1-2*loss1/loss_c); 2*loss1/loss_c = 1-sqrt(R0)
            
            print ("Peak number %d:" % (peak_number))
            print ("  Wavelength: %.2f +- %.2f" % (peak['x0'], peak['x0_err']))
            print ("      Height: %.2f +- %.2f" % (peak['ymax'], peak['ymax_err']))
            if 'Q' in peak.keys():
                print ("           Q: %.2f +- %.2f" % (peak['Q'], peak['Q_err']))
            print ("      Offset: %.2f +- %.2f" % (peak['yoffset'], peak['yoffset_err']))
            print ("          mc: {:.3}".format(mc))
            print (" cavity loss(%): {:.3}".format(loss_c*100))
            print ("coupler loss(%): {:.3}".format(loss_1*100))
            peak_number += 1


    def set_axes( self, axes ):
        """ this allows you to plot the data to a particular axes
        """
        self.ax = axes


    def peakdet(self, delta = 1, plotea = False):
        """
        find the dips on the spectrum
        delta is the amplitude in dB of the minimum detectable dip
        
        A point is considered a maximum peak if it has the maximal value,
        and was preceded (to the left) by a value lower by delta
        
        Converted from MATLAB script at http://billauer.co.il/peakdet.html
        Returns two arrays
        function [maxtab, mintab]=peakdet(v, delta, x)
        %PEAKDET Detect peaks in a vector
        %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
        %        maxima and minima ("peaks") in the vector V.
        %        MAXTAB and MINTAB consists of two columns. Column 1
        %        contains indices in V, and column 2 the found values.
        %      
        %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
        %        in MAXTAB and MINTAB are replaced with the corresponding
        %        X-values.
        % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
        % This function is released to the public domain; Any use is allowed.
        
        """
        maxtab = []
        mintab = []
        
        v = self.lum
        x = self.wavelen
           
        if x is None:
            x = arange(len(v))
        
        v = asarray(v)
        
        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')
        
        if not isscalar(delta):
            sys.exit('Input argument delta must be a scalar')
        
        if delta <= 0:
            sys.exit('Input argument delta must be positive')
        
        mn, mx = Inf, -Inf
        mnpos, mxpos = NaN, NaN
        
        lookformax = True
        
        for i in arange(len(v)):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]
            
            if lookformax:
                if this < mx-delta:
                    maxtab.append((mxpos, mx))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn+delta:
                    mintab.append((mnpos, mn))
                    mx = this
                    mxpos = x[i]
                    lookformax = True
        if plotea:
            self.plot()
            wdips = array(mintab)[:,0]
            pdips = array(mintab)[:,1]
            plot(wdips, pdips, 'o')
            
            L = 2*pi*13.84 #31.13e3
            fsr = diff(wdips)
            wav = array(mintab)[1:,0]
            ng = wav**2/(fsr)/L
            print (median(ng[argwhere(ng<5)]))
            
#            figure()
#            plot(wav,fsr);xlabel('Wavelength (nm)');ylabel('FSR')
#            figure()
#            plot(wav, ng);xlabel('Wavelength (nm)');ylabel('$n_g$')
    
        return array(mintab)[:,0] #return array(maxtab), array(mintab)

    def fit_fsr2(self, plotea = True):
        """Estimate fsr using peak detect
        
        """
        dips = self.peakdet(plotea = plotea)
        fsr = diff(dips)
        self.fsr = mean(fsr)
            
        return self.fsr


    def fit_fsr(self, sampling = 1e-12):
        """Estimate fsr using autocorrelation
        
        from https://gist.github.com/endolith/255291
        
        Pros: Best method for finding the true fundamental of any repeating wave, 
        even with strong harmonics or completely missing fundamental
        
        Cons: Not as accurate, doesn't work for inharmonic things like musical 
        instruments, this implementation has trouble with finding the true peak
        
        """
        signal = self.lum
        # Calculate autocorrelation (same thing as convolution, but with one input
        # reversed in time), and throw away the negative lags
        signal -= mean(signal) # Remove DC offset
        corr = fftconvolve(signal, signal[::-1], mode='full')
        corr = corr[len(corr)/2:]
                
        # Find the first low point
        d = diff(corr)
        start = find(d > 0)[0]
        
        # Find the next peak after the low point (other than 0 lag).  This bit is 
        # not reliable for long signals, due to the desired peak occurring between 
        # samples, and other peaks appearing higher.
        i_peak = argmax(corr[start:]) + start
        i_interp = parabolic(corr, i_peak)[0]
#        print "i_interp: %f" % i_interp
        self.fsr = i_interp * sampling*1e9
        return self.fsr
        

    def fit_resonances(self, dw = 0.8, delta = 1, printparams = False, plotfit = False, plot_coupler_vs_wavelength = False):
        """
        finds and fits all the resonances with a lorentzian fit
        
        dw is the wavelength span around the resonance
        delta
        plotfit: plots each fit in a separate figure
        plotfitw:         
        """
        self.fit_fsr2()
        wdips = self.peakdet(delta = delta)
        delta_c = zeros_like(wdips)
        delta_1 = zeros_like(wdips)
        Qs = zeros_like(wdips) 
        f0s = zeros_like(wdips) 
        
        for i in range(len(wdips)):
            wdip = wdips[i]
            wmin = wdip -0.5*dw
            wmax = wdip + 0.5*dw
            
            if wmax>max(self.w):
                wmax = max(self.w)

            if wmin<min(self.w):
                wmin = min(self.w)
            
            self.chop(wmin, wmax)
            self.fit_lorentzians(center = wdip, printparams=printparams, plotfit = False)
            
            if plotfit:
                figure()
                self.plot_fit( linewidth=2.0, fitrange=(0,-1), label = 'fit', lw=2, ls='--' )#  color='black'
            
            
            peak = self.fit_params[0]
            f0 = peak['x0']
            Q = peak['Q']
            R0 = 10**(-peak['ymax']/10.0) # convert to linear
#            print(peak['ymax'])
#            print(R0)
            mc = f0/self.fsr
            loss_c = 2*pi*mc/Q # cavity loss
            loss_1 = (1-sqrt(R0))*loss_c/2.0 #sqrt(R0) = (1-2*loss1/loss_c); 2*loss1/loss_c = 1-sqrt(R0)
            
            delta_c[i] = loss_c*100
            delta_1[i] = loss_1*100
            Qs[i] = Q
            f0s[i] = f0

            if plotfit:            
                plot(self.wavelen, self.lum, lw=2.0)
                title(' $\delta_c$ = {:.2}%, K = {:.2}%'.format(loss_c*100, loss_1*100))
                xlabel('Wavelength (nm)')
                ylabel('Transmission (dB)')
                legend(loc = 'best')
                savefig('./figures/'+ str(int(f0)))

            self.lum = array(self.p)
            self.wavelen = array(self.w)

        if plot_coupler_vs_wavelength:
            figure()
            plot(f0s, delta_1, '.')
            xlabel('Wavelength (nm)')
            ylabel('K1 (%)')
            figure()
            plot(f0s, delta_c, '.')
            xlabel('Wavelength (nm)')
            ylabel('RTL (%)')
            
#        print '%%%%%%%%%%%%%%%%%%%%%%  Averages  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print (" cavity loss(%): {:.3} +- {:.3}".format(mean(delta_c), std(delta_c)))
        print ("coupler loss(%): {:.3} +- {:.3}".format(mean(delta_1), std(delta_1)))
        return 0


    def fit_rings(self, dw = 0.4, delta = 2, printparams = False, plotfit = False, bs = True, plot_coupler_vs_wavelength = False):
        """
        finds and fits all the resonances fitting to the ring equation
        
        peakdet scans the peaks every delta wavelength span
        delta is the amplitude in dB of the minimum detectable dip
        chop, normalizes every peak with delta wavelength span
        
        bs = True  fits the rings with the backscattering model
        fits rings using a backscattering model
        dw is the wavelength span, centered at the resonance
        
        plotfit: plots each fit in a separate figure
        plotfitw: plots the cou
        
        finds and fits all the resonances
        dw is the wavelength span around the resonance
        
        """
        self.fit_fsr2()
        wdips = self.peakdet(delta = delta)
        delta_p = zeros_like(wdips)
        delta_1 = zeros_like(wdips)
        Qs = zeros_like(wdips) 
        f0s = zeros_like(wdips) 
        Rbs = zeros_like(wdips)
        
        for i in range(len(wdips)):
            wdip = wdips[i]
            wmin = wdip - 0.5*dw
            wmax = wdip + 0.5*dw
            
            if wmax>max(self.w):
                wmax = max(self.w)

            if wmin<min(self.w):
                wmin = min(self.w)
            
            [w, p] = self.chop(wmin, wmax)
            
            if bs: # back-scattering model = True
                p0 = array([1, 0.02, 35, wdip]) #  K = K2 = 1, RTL = 0.01, Rbs = 0.02
                popt, pcov = curve_fit(f_ring_bs, w, p, p0 = p0, bounds=(0, [100, 100, 100, wdip+10]))
                delta_p[i] = popt[1] #RTL
                delta_1[i] = popt[0]
                Rbs[i] = popt[2] # dB/mm
                delta_bs = 10**(-Rbs[i]/10)
#                print delta_bs
#                Rbs[i] = 10*log10(1e-2*popt[2]/(L_rt*1e-4)) # dB/cm

                if plotfit: 
                    figure()
                    plot(self.wavelen, self.lum, lw = 2)
                    plot(w,f_ring_bs(w, popt[0],popt[1],popt[2],popt[3]), label = 'fit', lw=2, ls='--')
                    
                    
#                    xlabel('Wavelength (nm)', fontsize = FONTX)
#                    ylabel('Transmission (dB)', fontsize = FONT)
                    xticks([wdip - 0.2, wdip - 0.1, wdip, wdip + 0.1, wdip + 0.2])
#                    title(' RTL = {:.2} %, K1 = {:.2} %, K2 = {:.2} % Rbs = {} dB'.format(delta_c[i], popt[0], popt[1], int(Rbs[i])  ))                   
#                    title('Tp = {:.2} %, K = {:.2} %, Rbs = {} dB/mm'.format(delta_c[i], delta_1[i], int(round(-Rbs[i]))  ))
#                    title('F = {:.2} ($\delta_c$ = {:.2}%), K1 = $\delta_1$ = {:.2} %, Rbs = {} dB/mm'.format(2*pi/delta_c[i],delta_c[i], delta_1[i], int(round(Rbs[i]))  ))
                    title('$\delta_c$ = {:.2}%, $\delta_1$ = {:.2} %, Rbs = {} dB'.format(delta_p[i] + 2*delta_1[i], delta_1[i], int(round(-Rbs[i]))))
  
                    legend()#loc = 'best'
                    plt.tight_layout()
                    if PDF:
                        savefig(FPATH + 'bs' + str(int(wdip))+'.pdf')
                    else:
                        savefig(FPATH + 'bs' + str(int(wdip)))
                    
                        

                
            else: # back-scattering model = False
                p0 = array([1, 0.1, wdip]) #  K1 = K2 = 1, RTL = 0.1
                popt, pcov = curve_fit(f_ring, w, p, p0 = p0, bounds=(0, [100, 100, wdip + 10]))
                delta_c[i] = popt[1] #RTL
                delta_1[i] = popt[0]   
                

                if plotfit:
                    figure()
                    plot(self.wavelen, self.lum, lw = 2)
                    plot(w,f_ring(w, popt[0],popt[1],popt[2]), label = 'fit', lw=2, ls='--')
#                    xlabel('Wavelength (nm)', fontsize = FONTX)
#                    ylabel('Transmission (dB)', fontsize = FONT)
                    xticks([wdip - 0.2, wdip - 0.1, wdip, wdip + 0.1, wdip + 0.2])
                    title('Tp = {:.2}%, K = {:.2}%'.format(delta_c[i], delta_1[i]))
                    legend() # loc = 'best'
                    plt.tight_layout()
                    if PDF:
                        savefig(FPATH + str(int(wdip))+'.pdf')
                    else:
                        savefig(FPATH + str(int(wdip)))
            

            self.reset_spectrum()

        if plot_coupler_vs_wavelength:
            figure()
            plot(f0s, delta_1, '.')
            xlabel('Wavelength (nm)')
            ylabel('K1 (%)')
            figure()
            plot(f0s, delta_p, '.')
            xlabel('Wavelengvth (nm)')
            ylabel('RLT (%)')
            
#        print '%%%%%%%%%%%%%%%%%%%%%%  Averages  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print (" cavity loss(%): {:.2} +- {:.2}".format(mean(delta_p), std(delta_p)))
        print ("coupler loss(%): {:.2} +- {:.2}".format(mean(delta_1), std(delta_1)))
        if bs:
            print ("Backscattering: {}dB/cm +- {}dB/cm".format( int(round(mean(Rbs))), int(round(std(Rbs)))) )
        return 0



"""
1x2 splitters 2015
"""
path = PATH + 'splitters/'
c = ['JoaquinMatres_SplitterCavityMMI1x2_1_1964', # 1
          'JoaquinMatres_SplitterCavityMMI1x2_1965', #0
#          'JoaquinMatres_AD_bent_g400_L0_2_2314',
          ] 

"""
Add drop rings 2015
"""

pRR5s= PATH + 'AD/'
RR5s =    ['JoaquinMatres_AD_Symm_g100_L0_2_1590', # symmetric
          'JoaquinMatres_AD_Symm_g200_L0_2_1586',
          'JoaquinMatres_AD_Symm_g300_L0_2_1582',
#          'JoaquinMatres_AD_Symm_g400_L0_2_1578',
#          'JoaquinMatres_AD_Symm_g500_L0_2_1574'
          ]


pRR5b= PATH + 'ADB/'
RR5b = ['JoaquinMatres_AD_bent_g200_L0_2_2338',# bent coupler, results backscatter paper
        'JoaquinMatres_AD_bent_g300_L0_2_2326',
        'JoaquinMatres_AD_bent_g400_L0_2_2314',
        ]


c = RR5b
path = pRR5b

FONT= 12
FONTX= 10

params = {
   'axes.labelsize': 12,
   'text.fontsize': 12,
   'legend.fontsize': 15,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'text.usetex': False,
   'figure.figsize': [4.5, 4.5]
   }
rcParams.update(params)
rcParams.update({'figure.autolayout': False})



path = './bs/'

#i = 0
#print(c[i])
#s = Spectrum(c[i], 1, path, remove_baseline =True)
#s.fit_rings(plotfit = True, bs = False)


figure()
i = 0
print(c[i])
s = Spectrum(c[i], 1, path, remove_baseline =True)
s.fit_rings(plotfit = True, bs = True)



#s.plot()
#bs = log(1- 10**(-25*0.1))
#p  = log(10**(1*0.1))
#Lc = 160*bs/(p**2)
#print(Lc*10) #mm
#
#bs = log(1- 10**(-25*0.1))
#p  = log(10**(7*0.1))
#Lc = 160*bs/(p**2)
#print(Lc*10) #mm
#
#bs = log(1- 10**(-25*0.1))
#p  = log(10**(11*0.1))
#Lc = 160*bs/(p**2)
#print(Lc*10) #mm


#s.plot()
#s.fit_resonances(plotfit = True)


#wav = linspace(1500,1600,1000)
#r = f_ring(wav, K1 = 1, K2 = 1, RTL = 0.1, wav0 = 1550)
#plot(wav, r)


RR5s = [
'JoaquinMatres_RR1s_w500_R5_g100_2_1376',
'JoaquinMatres_RR1s_w500_R5_g150_2_1374',
'JoaquinMatres_RR1s_w500_R5_g200_2_1372',
'JoaquinMatres_RR1s_w500_R5_g250_2_1370',
#'JoaquinMatres_RR1s_w500_R5_g300_2_1368',
]



#s = Spectrum(c[1], 1)
#s.fit_resonances(plotfit = True)
#s.fit_rings(plotfit = True)



#c = RR5s
#s = Spectrum(c[1], 1)
#s.fit_resonances(plotfit = True)
#s.fit_rings(plotfit = True)



#wdip = 1541.8
#dw = 0.5
#wmin = wdip - 0.5*dw
#wmax = wdip + 0.5*dw
#[w, p] = s.chop(wmin, wmax)
#
#p0 = array([1, 1, 0.01, 0.02, 0.6])
#popt, pcov = curve_fit(f_ring_bs, w, p, p0 = p0, bounds=(0, [100, 100, 100, 100, pi]))
#delta_c = popt[2] #RTL
#delta_1 = popt[0]
#Rbs = popt[3]
#
#plot(s.wavelen, s.lum, lw=2.0)
#plot(w,f_ring_bs(w, popt[0],popt[1],popt[2],popt[3], popt[4]), label = 'fit', lw=2.0, ls ='--')#linewidth=2.0, '--'
#xlabel('Wavelength (nm)')
#ylabel('Transmission (dB)')
#title(' $\delta_c$ = {:.2}%, K = {:.2}%, Rbs = {:.2}%'.format(delta_c, delta_1, Rbs))
#legend(loc = 'best')

#
#[w, p] = s.chop(1541, 1542.5)
#p = p-max(p)
#plot(w,p)
#s.plot()

#plot(w,f_ring_bs(w))

#p0 = array([1, 1, 0.01, 0.02, 0.6])
#popt = p0

#plot(w,f_ring_bs(w))
#popt, pcov = curve_fit(f_ring_bs, w, p, p0 = p0, bounds=(0, [100, 100, 100, 100, pi]))
#print(popt)
#
#plot(w,f_ring_bs(w, popt[0],popt[1],popt[2],popt[3], popt[4]))

#for i in range(len(c)):
#    s = Spectrum(c[i], 1);
#    s.plot(label = c[i][28:31])
#title('R = 5um, bend to bend')

#print c[1]
#s = Spectrum(c[1], 1)

#[w, p] = s.chop(1524, 1524.4)
#popt, pcov = curve_fit(f_ring, w, p)
#print popt
#print pcov


##[w, p] = s.chop(1513, 1535)
#p_fit = f_ring(w)
#
#plot(w,p_fit)
##[popt, pcov] = fit_ring(w, p)

#
##plot(w,p)
#s.plot('.')
#s.fit_resonances(plotfit = True)
#legend(loc = 'best')


#if __name__ == "__main__":        
#    test_spectrum()