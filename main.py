'''
Copyright (c) 2020, Sven-Arne Quist & Srikara Datta, Mothership Missions (Illegal Sand Mining Networks)
Distributed under the MIT License.
See accompanying file LICENSE.md or copy at http://opensource.org/licenses/MIT

Modified from kvos/coastsat by Kilian Vos WRL 2018
Authors: Sven-Arne Quist (https://github.com/S-AQ) & Srikara Datta (https://github.com/srikarad07)
'''

#!/usr/bin/env python
#==========================================================#
# Shoreline extraction from satellite images
#==========================================================#

# load modules
import os
import numpy as np
import pickle
import warnings
import geopandas as gpd
import sys
import json
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import random 

from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects


print("                                                                  ")
print("------------------------------------------------------------------")
print("                        Modified Coastsat                         ")
print("                              0.1.0                               ")
print("      Copyright (c) 2020, Sven-Arne Quist & Srikara Datta         ")
print("------------------------------------------------------------------")
print("")
print("")
print("******************************************************************")
print("                          Input parameters                        ")
print("******************************************************************")
print("")

# Read the inputs from the input file. 
inputFile = sys.argv[-1]

# Display the json file to read the inputs. 
print("Your inputs are being read from this file: ", inputFile)

# read file
with open(inputFile, 'r') as myfile:
    data=myfile.read()
    pass 

# parse file
inputParameters                     = json.loads(data)

# Get input parameters, #<int> comment represents the old order of reading each of these parameters
sitename                            = inputParameters["name"] #3
dates                               = inputParameters["dates"] #1
sat_list                            = inputParameters["satList"] #2    
inputGeoJsonFilePath                = inputParameters["studyAreaGeoJsonPath"] #4
dirtyHackFlag                       = inputParameters["dirtyHackFlag"] #6
outputEpsg                          = inputParameters["outputEpsg"] #10
checkDetection                      = inputParameters["checkDetection"] #11
saveFigure                          = inputParameters["saveFigure"] #12
colorStyle                          = inputParameters["colorStyle"] #13
cloudMaskIssueFlag                  = inputParameters["cloudMaskIssueFlag"] #5
sandColor                           = inputParameters["sandColor"] #7
referenceGeoJsonFiles               = inputParameters["referenceGeoJsonFiles"] #25
transectSettings                    = inputParameters["transectSettings"] #26
cloudThresh                         = inputParameters["cloudThresh"] #8
cloudThreshMonteCarlo               = inputParameters["cloudThreshMonteCarlo"] #9
minBeachArea                        = inputParameters["minBeachArea"] #14
minBeachAreaMonteCarlo              = inputParameters["minBeachAreaMonteCarlo"] #15
bufferSize                          = inputParameters["bufferSize"]  #16
bufferSizeMonteCarlo                = inputParameters["bufferSizeMonteCarlo"] #17 
minLengthSl                         = inputParameters["minLengthSl"] #18
minLengthSlMonteCarlo               = inputParameters["minLengthMonteCarlo"] #19 
maximumDistanceReference            = inputParameters["maximumDistanceReference"] #20 
maximumDistanceReferenceMonteCarlo  = inputParameters["maximumDistanceReferenceMonteCarlo"] #21
numberOfMonteCarloSimulations       = inputParameters["numberOfMonteCarloSimulations"] #22
videoOutputFlag                     = inputParameters["videoOutputFlag"] #23
saveJpgPreprocessFlag               = inputParameters["saveJpgPreprocessFlag"] #24

# Print the variables
print("Dates for analysis:                      ", dates)
print("List of satellites:                      ", sat_list)
print("Name of the site:                        ", sitename)
print("Input file with study area coordinates:  ", inputGeoJsonFilePath)
print("Is dirty hack flag on?                   ", dirtyHackFlag)

print("")
print("******************************************************************")
print("                   End of input parameters                        ")
print("******************************************************************")
print("")


print("")
print("******************************************************************")
print("                Generating the input polygon                      ")
print("******************************************************************")
print("")

# Region of interest (Geojson file, can be retrieved from google Earth engine) 
aoi_poly                = gpd.read_file(inputGeoJsonFilePath)
g                       = [i for i in aoi_poly.geometry]
x,y                     = g[0].exterior.coords.xy
coords                  = np.dstack((x,y)).tolist()
polygon                 = coords 
print("Generated polygon coordinates:           ", polygon)

# filepath where data will be stored
filepath_data           = os.path.join(os.getcwd(), 'data')

# put all the inputs into a dictionnary
inputs = {
    'polygon': polygon,
    'dates': dates,
    'sat_list': sat_list,
    'sitename': sitename,
    'filepath': filepath_data
        }

## Check if the path exists or not
path_to_check           = os.path.join(filepath_data, sitename)

print("")
print("******************************************************************")
print("                Preparing the dataset                             ")
print("******************************************************************")
print("")

if os.path.exists(path_to_check) == False: 
    # before downloading the images, check how many images are available for your inputs
    SDS_download.check_images_available(inputs)

    #%% 2. Retrieve images
    # only uncomment this line if you want Landsat Tier 2 images (not suitable for time-series analysis)
    # inputs['include_T2'] = True 
    # retrieve satellite images from GEE    
    metadata            = SDS_download.retrieve_images(inputs)
else:
    print("You have downloaded files in the your directory! Let's reuse them!") 
    pass 

# Download the metadata fromt the input files
metadata                = SDS_download.get_metadata(inputs) 

settings = { 
    # Temp flag for dirty hack 
    'dirty_hack_flag': dirtyHackFlag,
    'videoOutputFlag': videoOutputFlag,
    "referenceGeoJsonFiles": referenceGeoJsonFiles,
    # general parameters:
    'cloud_thresh': cloudThresh, # threshold on maximum cloud cover
    'output_epsg': outputEpsg,  # epsg code of spatial reference system desired for the output   
    # quality control:
    'check_detection': checkDetection, # if True, shows each shoreline detection to the user for validation
    'save_figure': saveFigure,  # if True, saves a figure showing the mapped shoreline for each image
    'color_style': colorStyle, # if True, saves figure as true color image. If False, saves figure as false color image. 
    # add the inputs defined previously
    'inputs': inputs,
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    'min_beach_area': minBeachArea,     # minimum area (in metres^2) for an object to be labelled as a beach
    'buffer_size': bufferSize,         # radius (in metres) of the buffer around sandy pixels considered in the shoreline detection
    'min_length_sl': minLengthSl,       # minimum length (in metres) of shoreline perimeter to be valid
    'cloud_mask_issue': cloudMaskIssueFlag,  # switch this parameter to True if sand pixels are masked (in black) on many images  
    'sand_color': sandColor,    # 'default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
}

# save jpeg of processed satellite image
if saveJpgPreprocessFlag:
    SDS_preprocess.save_jpg(metadata, settings)
    pass 

print("")
print("******************************************************************")
print("                Loading the shoreline                          ")
print("******************************************************************")
print("")

SDS_tools.generateReferencePklFile(settings)
settings['reference_shoreline']     = SDS_preprocess.get_reference_sl(metadata, settings)
settings['max_dist_ref']            = maximumDistanceReference

print("")
print("******************************************************************")
print("                Extracting the shoreline                          ")
print("******************************************************************")
print("")

# Moment of truth
if numberOfMonteCarloSimulations == 0: 
    output = SDS_shoreline.extract_shorelines(metadata, settings)
else:
    print("Running the Monte Carlo simulations!")
    for ii in range(numberOfMonteCarloSimulations):
        if minBeachAreaMonteCarlo != None: 
            settings["min_beach_area"] = random.randint(
                    (minBeachArea - minBeachAreaMonteCarlo), (minBeachArea + minBeachAreaMonteCarlo))
            pass 
        if bufferSizeMonteCarlo != None:
            settings["buffer_size"] = random.randint(
                    (bufferSize - bufferSizeMonteCarlo), (bufferSize + bufferSizeMonteCarlo))
            pass 
        if minLengthSlMonteCarlo != None:
            settings["min_length_sl"] = random.randint(
                    (minLengthSl - minLengthSlMonteCarlo), (minLengthSl + minLengthSlMonteCarlo))
            pass 
        if maximumDistanceReferenceMonteCarlo != None:
            # max distance (in meters) allowed from the reference shoreline
            settings['max_dist_ref'] = random.randint(
                    (maximumDistanceReference - maximumDistanceReferenceMonteCarlo), (maximumDistanceReference + maximumDistanceReferenceMonteCarlo)) 
            pass
        output = SDS_shoreline.extract_shorelines(metadata, settings)
        SDS_tools.make_video(settings)

        # transectSettings['additional inputs']       = settings
        # transects_gdf = SDS_transects.auto_comp_transects(transectSettings)
        # transects_gdf.plot()
        # transect_extension = str(settings["min_beach_area"])
        # transect_extension += "_"
        # transect_extension += str(settings["buffer_size"])
        # transect_extension += "_"
        # transect_extension += str(settings["min_length_sl"])
        # transect_extension += "_"
        # transect_extension += str(settings["max_dist_ref"])
        # transect_extension += "_transects.geojson"

        # geojson_file = os.path.join(os.getcwd(), inputs['sitename'], transect_extension)
        # transects = SDS_tools.transects_from_geojson(geojson_file)
        
        pass 

print("")
print("******************************************************************")
print("             Shorelines have been extracted                       ")
print("******************************************************************")
print("")


# To plot the shorelines from the output. 
# <<< NOT NEEDEED since it is being done in post-processing.
fig = plt.figure()
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
for i in range(len(output['shorelines'])):
    sl = output['shorelines'][i]
    date = output['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'))
plt.legend()
mng = plt.get_current_fig_manager()                                         
mng.window.showMaximized()    
fig.set_size_inches([15.76,  8.52])

