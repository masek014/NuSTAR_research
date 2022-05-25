#!/bin/python3

import datetime
import sunpy.map
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.patches as patches # for rectangles in Sunpy V3.1.0, I can't get draw_rectangle() to work
import numpy as np

from reproject import reproject_interp
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from sunpy.coordinates import Helioprojective, RotatedSunFrame, transform_with_sun_center
from sunpy.net import Fido, attrs as a
from aiapy.calibrate import register, update_pointing, normalize_exposure

import warnings
warnings.filterwarnings("ignore")


# Define constants
X_LABEL = 'X (arcseconds)'
Y_LABEL = 'Y (arcseconds)'
AIA_WAVELENGTH = 94 # Units of angstroms
STEREO_MIN_WAVELENGTH = 171 # Units of angstroms
STEREO_MAX_WAVELENGTH = 195 # Units of angstroms

# Define the universal axes limits to be used by all plots.
# In units of arcseconds.
X_MIN = -1100
Y_MIN = -1100
X_MAX = 1100
Y_MAX = 1100

def add_minor_ticks(ax):
    """
    Adds minor ticks to the plot on the provided axes.
    """

    (ax.coords[0]).display_minor_ticks(True)
    (ax.coords[0]).set_minor_frequency(5)
    (ax.coords[1]).display_minor_ticks(True)
    (ax.coords[1]).set_minor_frequency(5)
    ax.tick_params(which='minor', length=1.5)

def most_recent_map(_map):
    """
    Query and return the most recent _map (of "AIA" or "STEREO") 
    that Sunpy.Fido can get.
    
    Parameters
    ----------
    _map : string
        The instrument you want the most recent map from. 
        E.g., _map="aia" or _map="stereo". 
        
    Returns
    -------
    Sunpy generic map.
    """
    
    if _map.lower()=="aia":
        info = (a.Instrument("aia") & a.Wavelength(AIA_WAVELENGTH*u.angstrom))
        look_back = {"minutes":30}
    elif _map.lower()=="stereo":
        info = (a.Source('STEREO_A') & a.Instrument("EUVI") & \
            a.Wavelength(STEREO_MIN_WAVELENGTH*u.angstrom, STEREO_MAX_WAVELENGTH*u.angstrom)) 
        look_back = {"days":5}
    else:
        print("Don't know what to do! Please set _map=\"aia\" or \"stereo\".")
        
    current = datetime.datetime.now()
    current_date = current.strftime("%Y-%m-%dT%H:%M:%S")    

    past = current-datetime.timedelta(**look_back)
    past_date = past.strftime("%Y-%m-%dT%H:%M:%S")
    startt = str(past_date)
    endt = str(current_date)

    result = Fido.search(a.Time(startt, endt), info)
    file_download = Fido.fetch(result[0, -1], site='ROB')

    data_map = sunpy.map.Map(file_download[-1])

    # Set the axes limits.
    bl = SkyCoord(X_MIN*u.arcsec, Y_MIN*u.arcsec, frame=data_map.coordinate_frame)
    tr = SkyCoord(X_MAX*u.arcsec, Y_MAX*u.arcsec, frame=data_map.coordinate_frame)
    data_map = data_map.submap(bottom_left=bl, top_right=tr)
    
    return data_map 


def plot_map(in_map, ax, cmap='gray', title=''):
    """
    Plot the provided map on the provided axes.

    Parameters
    ----------
    in_map : Sunpy map
        The input map to be projected.
    ax : matplotlib.pyplot axes object
        The axes the NuSTAR fov is to be drawn on.
    cmap : str
        The matplotlib colormap to use on the scale.
    title : str
        The title of the plot.
    
    Returns
    -------
    None
    """
    
    in_map.plot(cmap=cmap, title=title)
    in_map.draw_limb(color='black')
    ax.tick_params(which='major', direction='in')
    ax.grid(False)
    ax.set(xlabel=X_LABEL, ylabel=Y_LABEL)
    add_minor_ticks(ax)
    plt.colorbar(fraction=0.046, pad=0.04)
    

def project_map(in_map, future_time):
    """
    Create a projection of an input map at the given input time.
    
    Parameters
    ----------
    in_map : Sunpy map
        The input map to be projected.
    future_time : str
        The time of the projected map. 
        
    Returns
    -------
    Sunpy generic map.
    """

    in_time = in_map.date
    out_frame = Helioprojective(observer='earth', obstime=future_time,
                                rsun=in_map.coordinate_frame.rsun)
    rot_frame = RotatedSunFrame(base=out_frame, rotated_time=in_time)

    out_shape = in_map.data.shape
    out_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=out_frame)
    header = sunpy.map.make_fitswcs_header(out_shape,
                                           out_center,
                                           scale=u.Quantity(in_map.scale))
    
    out_wcs = WCS(header)
    out_wcs.coordinate_frame = rot_frame

    with transform_with_sun_center():
        arr, _ = reproject_interp(in_map, out_wcs, out_shape)
    
    out_warp = sunpy.map.Map(arr, out_wcs)
    out_warp.plot_settings = in_map.plot_settings

    # Set the axes limits.
    bl = SkyCoord(X_MIN*u.arcsec, Y_MIN*u.arcsec, frame=out_warp.coordinate_frame)
    tr = SkyCoord(X_MAX*u.arcsec, Y_MAX*u.arcsec, frame=out_warp.coordinate_frame)
    
    out_warp = out_warp.submap(bottom_left=bl, top_right=tr)

    return out_warp


def draw_nustar_fov(in_map, ax, center_x, center_y, layers=[-100, 0, 100], colors='red', angle=0, pixscale=None):
    """
    Draw squares representing NuSTAR's field of view on the current map.
    
    By default, three squares are drawn: one that is equal
    to the 12x12 arcminute FOV and two with side lengths
    of +-100 arcseconds from the actual side lengths.
    
    Parameters
    ----------
    in_map : Sunpy map
        The input map on which the squares will be overlaid.
    ax : matplotlib.pyplot axes object
        The axes the NuSTAR fov is to be drawn on.
    center_x : float
        The x position, in arcseconds, of the squares' center point.
    center_y : float
        The y position, in arcseconds, of the squares' center point.
    layers : list
        List of values, in arcseconds, containing the adjustments
        to the side lengths of the drawn squares. Each value results
        in a new square drawn on the map.
    colors : str or list of str
        The colors of the drawn squares. If colors is a string,
        then each square will be drawn with that color.
        Otherwise, a list can be provided to customize the color
        of each layer. The index of the color will match the index
        of the layer.
    angle : int or float
        Anti-clockwise rotation from Solar north for NuSTAR field of view.
    pixscale : float
        Arcsecond-to-pixel conversion for the original AIA or STEREO map.
        Needed for the NuSTAR field of view rotation.
        
    Returns
    -------
    None
    """
    
    # Change the colors variable to a list if it's not already one.
    if not isinstance(colors, list):
        colors = [colors]*len(layers)
    
    FOV_SIDE_LENGTH = 12*60 # Units of arcseconds

    angle_str = ""
    for i, diff in enumerate(layers):
        if angle==0:
            # Translate to bottom left corner of rectangle
            bottom_left = ((center_x - FOV_SIDE_LENGTH/2 - diff)*u.arcsec,
                           (center_y - FOV_SIDE_LENGTH/2 - diff)*u.arcsec)
            rect_bl = SkyCoord(*bottom_left, frame=in_map.coordinate_frame)

            in_map.draw_quadrangle(bottom_left=rect_bl,
                                     width=(FOV_SIDE_LENGTH+2*diff)*u.arcsec,
                                     height=(FOV_SIDE_LENGTH+2*diff)*u.arcsec,
                                     color=colors[i],
                                     axes=ax)
        else:
            # **kwargs dont get passed to matplotlib so not easy way to rotate, do it myself
            # get boxes in pixels, newer Sunpy doesn't allow draw_rectangle here
            center_pix = [in_map.data.shape[1]/2, in_map.data.shape[0]/2]
            
            # get bottom left coords in ref. frame where center of box 
            bx_arc_square, by_arc_square = -FOV_SIDE_LENGTH/2 - diff, -FOV_SIDE_LENGTH/2 - diff
            
            # rotate bottom left clockwise to find where box to-be-rotated to maintain centers needs to be
            rot_mat = np.array([[np.cos(-angle*(np.pi/180)), np.sin(-angle*(np.pi/180))],
                                [-np.sin(-angle*(np.pi/180)), np.cos(-angle*(np.pi/180))]]) @ np.array([bx_arc_square,by_arc_square])

            bx_rotarc, by_rotarc = rot_mat[0], rot_mat[1]
            
            # bottom left of new box in arcsec where cneter of the Sun is (0,0)
            bx_arc, by_arc = (center_x+bx_rotarc), (center_y+by_rotarc)
            
            # create and rotate box
            rect = patches.Rectangle([center_pix[0]+bx_arc/pixscale,
                                      center_pix[1]+by_arc/pixscale], 
                                     (FOV_SIDE_LENGTH+2*diff)/pixscale, 
                                     (FOV_SIDE_LENGTH+2*diff)/pixscale, 
                                     facecolor="none", 
                                     linewidth=2, 
                                     edgecolor=colors[i], 
                                     angle=angle)
            angle_str = "Rotated "+str(angle)+" deg Anti-clockwise"
            ax.add_patch(rect)

        # Add text on plot.
        # Determine the position of the text box.
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        x_mid, y_mid = (X_MAX+X_MIN)/2, (Y_MAX+Y_MIN)/2
        text_x = ax_xlim[1] * (1-(x_mid-X_MIN)/(X_MAX-X_MIN))
        # if the fov centre is in the lower half then put text in the top and vice versa
        text_y = ax_ylim[1] * (y_mid-Y_MIN-900)/(Y_MAX-Y_MIN) if center_y>=0 else ax_ylim[1] * (1-(y_mid-Y_MIN-900)/(Y_MAX-Y_MIN))
        
        # To make the text dynamic, we need to format
        # the text string based on the layers list.
        layers_copy = layers.copy()
        if 0 in layers_copy:
            layers_copy.remove(0)

        # Convert the list of int to list of str, add arcsecond unit symbols,
        # and add '+' to positive numbers
        list_of_strings = ['+'+str(x)+'\"' if x>0 else str(x)+'\"' for x in layers_copy]

        # Format the string by removing the residual brackets and quotes.
        layers_str = str(list_of_strings).replace('[','').replace(']','').replace('\'','')
        text_str = 'Center: ('+str(center_x)+'\",'+str(center_y)+'\")'+'\nBoxes 12\', ' + layers_str +"\n"+ angle_str
        
        if len(layers)>0:
            # Add the text to the plot.
            ax.text(text_x, text_y, text_str, color='red',
                     horizontalalignment='center', verticalalignment='center')


def points_of_interest_marker(x, y, ax, frame):
    """
    Plots points of interest on a sunpy map.
    
    Parameters
    ----------
    x,y : list
        List of the x and y coordinates of the points of interest.
    ax : matplotlib.pyplot axes object
        The axes to be plotted on.
    frame : sunpy map
        Coordinate frame being used.
        
    Returns
    -------
    None.
    """

    x_poi = x * u.arcsec # points of interest
    y_poi = y * u.arcsec 
    coords = SkyCoord(x_poi, y_poi, frame=frame.coordinate_frame)
    p = ax.plot_coord(coords, 'x', color="r") # plot points of interest


def plot_psp_loc(x, y, ax, frame):
    """
    Plots a square to represent PSP's location.
    
    Parameters
    ----------
    x,y : list
        List of the x and y coordinates of the points of interest.
    ax : matplotlib.pyplot axes object
        The axes to be plotted on.
    frame : sunpy map
        Coordinate frame being used.
        
    Returns
    -------
    None.
    """
    x_poi = x * u.arcsec # points of interest
    y_poi = y * u.arcsec 
    coords = SkyCoord(x_poi, y_poi, frame=frame.coordinate_frame)
    p = ax.plot_coord(coords, 's', color="grey") # plot points of interest

    # Determine the position of the text box.
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()
    x_mid, y_mid = (X_MAX+X_MIN)/2, (Y_MAX+Y_MIN)/2
    text_x = ax_xlim[1] * (x_mid-X_MIN+x)/(X_MAX-X_MIN)
    text_y = ax_ylim[1] * (y_mid-Y_MIN+y)/(Y_MAX-Y_MIN) 
    ax.text(text_x, text_y, "PSP", color='k',
                     horizontalalignment='center', verticalalignment='top', size=8)


def reprojection(obstime:str, center=None, layers=None, angle=0, psp_loc=None, markers=None):
    """
    Creates plot of the AIA and STEREO plot of a specific time projected onto a future time.
    
    Parameters
    ----------
    obstime : string
        Time the AIA and STEREO maps should be projected to; e.g., '2021-11-10T12:00:00'.
    center : list
        A coordinate pair (x,y) marking the center of the FOV.
    layers : list
        List of values, in arcseconds, containing the adjustments
        to the side lengths of the drawn squares. Each value results
        in a new square drawn on the map.
    angle : int or float
        Anti-clockwise rotation from Solar north for NuSTAR field of view.
    psp_loc : list
        List of [x,y] coordinates, in arcseconds, of PSP to be marked on the map. 
        E.g., [[0,100], [1000,500]] would mark two points, one at (0,100) and another (1000,500).
    markers : list
        List of [x,y] coordinate pairs, in arcseconds, to mark points of interest on the disk.
        
    Returns
    -------
    Tuple of axes for each subplot created.
    """

    # For AIA.
    aiamap = most_recent_map(_map="aia")
    projected_aiamap = project_map(aiamap, obstime)
    reversed_aia_cmap = (aiamap.cmap).reversed()
    print("Got AIA map.")

    # Original AIA map.
    ax1 = plt.subplot(2, 2, 1, projection=aiamap)
    plot_map(aiamap, ax1, reversed_aia_cmap, title=f"Original AIA Map\nAIA " + \
        str(AIA_WAVELENGTH) + f" {aiamap.date}")

    # Projected AIA map.
    ax2 = plt.subplot(2, 2, 2, projection=projected_aiamap)
    plot_map(projected_aiamap, ax2, reversed_aia_cmap, title="Reprojected to an Earth Observer\nAIA " + \
        str(AIA_WAVELENGTH) + f" {projected_aiamap.date}")

    # For STEREO.
    stereomap = most_recent_map(_map="stereo")
    projected_stereomap = project_map(stereomap, obstime)
    reversed_stereo_cmap = (stereomap.cmap).reversed()
    print("Got STEREO-A map.")

    # Original STEREO map.
    ax3 = plt.subplot(2, 2, 3, projection=stereomap)
    plot_map(stereomap, ax3, reversed_stereo_cmap, title=f"Original STEREO Map\nSTEREO " + \
        str(STEREO_MIN_WAVELENGTH) + "-" + str(STEREO_MAX_WAVELENGTH) + f" {stereomap.date}")
    plt.set_cmap('YlGn')

    # Projected STEREO map.
    ax4 = plt.subplot(2, 2, 4, projection=projected_stereomap)
    plot_map(projected_stereomap, ax4, reversed_stereo_cmap, title="Reprojected to an Earth Observer\nSTEREO " + \
        str(STEREO_MIN_WAVELENGTH) + "-" + str(STEREO_MAX_WAVELENGTH) + f" {projected_stereomap.date}")
    plt.set_cmap('YlGn')

    if center is not None:
        draw_nustar_fov(projected_aiamap, ax2, *center, layers,
            angle=angle, pixscale=u.Quantity(aiamap.scale).value[0])
        draw_nustar_fov(projected_stereomap, ax4, *center, layers,
            angle=angle, pixscale=u.Quantity(stereomap.scale).value[0])

    if psp_loc is not None:
        for coord in psp_loc:
            plot_psp_loc(*coord, ax=ax2, frame=projected_aiamap)
            plot_psp_loc(*coord, ax=ax4, frame=projected_stereomap)
    if markers is not None:
        for coord in markers:
            points_of_interest_marker(*coord, ax=ax2, frame=projected_aiamap)
            points_of_interest_marker(*coord, ax=ax4, frame=projected_stereomap)

    plt.subplots_adjust(wspace=0.3, hspace=0.18)

    return (ax1, ax2, ax3, ax4)