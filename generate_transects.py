#-------------------------------------------------------------------------------
# Name:        perp_lines.py
# Purpose:     Generates multiple profile lines perpendicular to an input line
#
# Author:      JamesS
# Found on: https://gis.stackexchange.com/questions/50108/elevation-profile-10-km-each-side-of-a-line/50841#50841
# 
# Created:     13/02/2013
# 
# Contributor: Sven-Arne Quist
# Updated on:  28-Apr-2020
#-------------------------------------------------------------------------------
""" Takes a shapefile containing a single line as input. Generates lines
    perpendicular to the original with the specified length and spacing and
    writes them to a new shapefile.

    The data should be in a projected co-ordinate system.
"""

import numpy as np
import pandas as pd
from fiona import collection
from shapely.geometry import LineString, MultiLineString, Point
import geopandas as gpd


my_transect_settings = {
    'filename':  r'C:\Users\Administrator\Downloads\CoastSat\poyang_refline1.geojson',
    'spacing': 50,
    'section_length': 800,
    'output_EPSG': 32650,
    'sitename' : "POYANG",
    'filepath' : r'C:\Users\Administrator\Downloads\CoastSat\data',
    
    #'additional inputs': settings
    }

def auto_comp_transects(transect_settings):
    """
    Automatically generates transects along a line given as geojson. 
    Output is written to geojson as well. 
    
    Credits: Sven-Arne Quist & JamesS
    Date : 5-May-2020
    
    Arguments:
    -----------
    transect_settings: dict with the following keys
        'filename' : str
            filename of the reference shoreline
        'spacing' : int 
            Profile spacing. The distance at which to space the perpendicular profiles
            In the same units as the original shapefile (e.g. metres)
        'section_length': int
             Length of cross-sections to calculate either side of central line
             i.e. the total length will be twice the value entered here.
        'addtional inputs': dict
            contents of dictionary settings in which general settings for the model are recorded
            contains: 'output_epsg': int with coordinate system defined as epsg code. 
        
    Returns:    
    -----------
    all_transects: GeoDataFrame 
        GeoDataFrame containing LineStrings with all transects perpendicular to the reference line.
    
    """
    spc = transect_settings['spacing']
    sect_len = transect_settings['section_length']
    output_epsg_no = transect_settings['output_EPSG'] #settings['output_epsg']
    sitename = transect_settings['sitename'] #settings['inputs']['sitename']
    filepath = transect_settings['filepath'] #os.path.join(settings['inputs']['filepath'], sitename)
    
    # read reference shoreline in 'refline'
    refline = gpd.read_file(transect_settings['filename'])
    
    # check for projected coordinates
    wgs84 = {'init': 'epsg:4326'}
    if refline.crs != wgs84:
        # if not in wgs84, reproject to get projected coordinates
        refline.crs = wgs84
    
    # convert refline into LineString to used in the forloop 
    line = LineString(refline.geometry[0])
    
    # prepare empty geometry list
    geometry = []

    # prepare emtpy transect ID dictionary     
    transect_id = {"name" : []}
    
    # Calculate the number of profiles to generate
    n_prof = int(line.length/spc)
    
    """ 
    The main structure of this forloop is credited to JamesS 
    and was found on https://gis.stackexchange.com/questions/50108/elevation-profile-10-km-each-side-of-a-line/50841#50841
    """
    
    # Start iterating along the line
    for prof in range(1, n_prof+1):
        # Get the start, mid and end points for this segment
        seg_st = line.interpolate((prof-1)*spc)
        seg_mid = line.interpolate((prof-0.5)*spc)
        seg_end = line.interpolate(prof*spc)
    
        # Get a displacement vector for this segment
        vec = np.array([[seg_end.x - seg_st.x,], [seg_end.y - seg_st.y,]])
    
        # Rotate the vector 90 deg clockwise and 90 deg counter clockwise
        rot_anti = np.array([[0, -1], [1, 0]])
        rot_clock = np.array([[0, 1], [-1, 0]])
        vec_anti = np.dot(rot_anti, vec)
        vec_clock = np.dot(rot_clock, vec)
    
        # Normalise the perpendicular vectors
        len_anti = ((vec_anti**2).sum())**0.5
        vec_anti = vec_anti/len_anti
        len_clock = ((vec_clock**2).sum())**0.5
        vec_clock = vec_clock/len_clock
    
        # Scale them up to the profile length
        vec_anti = vec_anti*sect_len
        vec_clock = vec_clock*sect_len
    
        # Calculate displacements from midpoint
        prof_st = (seg_mid.x + float(vec_anti[0]), seg_mid.y + float(vec_anti[1]))
        prof_end = (seg_mid.x + float(vec_clock[0]), seg_mid.y + float(vec_clock[1]))
        
        # Connect starting point and end point of transect into a LineString
        prof_line = LineString([prof_st, prof_end])
        
        # append line to geometry list 
        geometry.append(prof_line)
        
        # append id to 
        transect_id["name"].append(str(prof))
    
    # group all transects into a single GeoDataFrame 
    all_transects = gpd.GeoDataFrame(transect_id, geometry=geometry)
    
    # reproject into output coordinate system
    outcrs = {'init': 'epsg:'+str(output_epsg_no)}
    all_transects.crs = outcrs
    
    # write to file
    out_name = filepath + sitename + '_auto_transects.geojson'
    all_transects.to_file(filename = out_name, driver='GeoJSON', encoding='utf-8')
    print("Transects are written to file")
    
    return(all_transects)

my_transect = auto_comp_transects(my_transect_settings) 
my_transect.plot()   
