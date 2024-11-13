from pathlib import Path
from dask import array as da
import seaborn as sns
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import pandas as pd




class load_data:
    # Attributes
    baseResolution = [1.8, 1.8, 2] # microns
    zarrMultiple = {j : 2 ** j for j in range(5)} # compression at each zarr level
    
    # Initiator
    def __init__(self, sample, level = 3):
        self.sample = str(sample)
        self.getPath()
        self.setLevel(level)
        self.setColorMaps()
        
    # Methods
    def getPath(self):
        # Method to get path to whole brain volume data
        rootDir = Path('../data')
        rootDir = [file for file in rootDir.iterdir() if self.sample in str(file)]
        
        # Check that the appropriate number of folders were found.
        if len(rootDir) > 1:
            raise ValueError("Found multiple directories matching requested sample ID.")
        elif len(rootDir) == 0:
            raise ValueError("Could not find a data directory matching input sample ID.")
        self.rootDir = rootDir[0]
            
        # Handle iteration of several formatting conventions
        sampleDir = rootDir[0].joinpath('processed', 'stitching', 'OMEZarr')
        if not sampleDir.exists():
            sampleDir = rootDir[0].joinpath('processed', 'OMEZarr')
            if not sampleDir.exists():
                sampleDir = rootDir[0].joinpath("image_tile_fusing","OMEZarr")
        print(f"Loading data from {sampleDir}")
        
        # Grab channel, named by excitation
        chPaths = {exCh.name.split('_')[1]: exCh for exCh in sampleDir.glob('Ex*.zarr')}
        self.channels = list(chPaths.keys())
        self.chPaths = chPaths
        print(f"Found the following channels: {self.channels}")
        
        # Grab cell segmentations
        segDir = rootDir[0].joinpath("image_cell_segmentation")
        segPaths = {exCh.name.split('_')[1]: exCh.joinpath("detected_cells.xml") for exCh in segDir.glob('Ex*')}
        self.segPaths = segPaths
        print(f"Found cell segmentations in the following channels: {list(segPaths.keys())}")
        
        # Grab CCF quantifications
        quantDir = rootDir[0].joinpath("image_cell_quantification")
        quantPaths = {exCh.name.split('_')[1]: exCh.joinpath("cell_count_by_region.csv") for exCh in quantDir.glob('Ex*')}
        ccfCellsPaths = {exCh.name.split('_')[1]: exCh.joinpath("transformed_cells.xml") for exCh in quantDir.glob('Ex*')}
        self.quantPaths = quantPaths
        self.ccfCellsPaths = ccfCellsPaths
        print(f"Found CCF aligned quantifications in the following channels: {list(quantPaths.keys())}")
        
    def setLevel(self,level,printOutput = True):
        # Method to update level and grab hierarchical volume for corresponding resolution level
        self.level = level
        if printOutput:
            print(f"Grabbing volumes for level: {level}")
        self.getVol()
    
    def getVol(self):
        # Method to mount volumetric imaging data
        self.vols = {channel: da.from_zarr(str(chPath), self.level).squeeze() for channel,chPath in self.chPaths.items()}
    
    def orientVol(self, ch, plane = "coronal", returnLabels = False):
        # Method to orient requested channel volume to a particular plane. Return labels for internal methods, e.g. plotSlice
        if (plane.lower() == "horizontal") | (plane.lower() == "transverse"):
            printTxt = "Plotting horizontal axis, "
            axis = 0
            xLabel = "M/L"
            yLabel = "A/P"
        elif plane.lower() == "sagittal":
            printTxt = "Plotting sagittal axis, "
            axis = 2
            xLabel = "A/P"
            yLabel = "D/V"
        else:
            plane = "coronal"
            printTxt = "Plotting coronal axis, "
            axis = 1
            xLabel = "M/L"
            yLabel = "D/V"
        chVol = da.moveaxis(self.vols[ch],axis,0)
        
        if returnLabels:
            return chVol, xLabel, yLabel, printTxt
        else:
            return chVol
    
    def setColorMaps(self, base = "black",channelColors = {}):
        # Method to establish each channel's color map for future plotting. Modifies default colors via channelColors channel:color dictionary pairs
        colorSets = {"445":"turquoise","488":"lightgreen","561":"tomato","639":"white"} # default colors
        colormaps = {}
        
        # Modify color sets if channel colors are provided
        for ch, color in channelColors.items():
            if ch not in self.channels:
                raise ValueError(f"Trying to set color for channel {ch}, but channel was not found in dataset.")
            else:
                colorSets[ch] = color
        
        # Generate color maps for channels present in data
        for ch in self.channels:
            if ch not in colorSets.keys():
                print(f"No default color exists for the {ch} channel, setting to white.")
                colormaps[ch] = sns.blend_palette([base,'white'], as_cmap = True)
            else:
                colormaps[ch] = sns.blend_palette([base,colorSets[ch]], as_cmap = True)
        self.colormaps = colormaps
        
    def plotSlice(self, ch = [], plane = "coronal", section = [], extent = [], level = 3, ticks = True, printOutput = True):
        """ 
        Plots a single brain slice from the volumetric data in a specified plane.

        Parameters
        ----------
        ch : str, optional
            Imaging channel to plot (e.g., "488", "561"). If not specified, defaults to the shortest wavelength available.
        plane : str, optional
            Plane in which to view the slice: "coronal", "sagittal", or "transverse". Defaults to "coronal".
        section : int or float, optional
            Position along the selected plane, in microns, to slice. If not specified, defaults to the midpoint.
        extent : list of float, optional
            4-element list defining the [left, right, bottom, top] extent of the plot in microns.
            If not provided, defaults to the entire image field.
        level : int, optional
            Downsampling level for the data. Higher levels correspond to more downsampling, for faster plotting. Default is 3.
        ticks : bool, optional
            Whether to display x and y axis labels and tick marks. Default is True.
        printOutput : bool, optional
            If True, prints information about the section and level being plotted. Default is True.

        Returns
        -------
        None
            Displays a matplotlib plot of the specified slice.
        
        """ 
                
        # If no channel is provided, plot shortest wavelength
        if not ch:
            ch = min(self.channels)
        
        # Specify resolution level, and then retrieve properly oriented volume
        self.setLevel(level, printOutput)
        [chVol, xLabel, yLabel, printTxt] = self.orientVol(ch, plane = plane, returnLabels = True)
        
        # Get data indices to be plotted
        if not section:
            sectionIndex = int(chVol.shape[0] / 2)
            section = sectionIndex * self.zarrMultiple[level]
        else: # otherwise convert microns to indices
            sectionIndex = int(section / self.zarrMultiple[level])
        if (not extent) | len(extent) != 4:
            extentIndices = np.array([0, chVol.shape[2], chVol.shape[1], 0])
            extent = extentIndices*self.zarrMultiple[level]
        else: #interpret extent requests as microns, convert to indices
            extentIndices = np.round(np.array(extent) / self.zarrMultiple[level])
        if printOutput:
            print(printTxt + 'section: ' + str(section) + ' (level ' + str(level) + ' index: ' + str(sectionIndex) + ')')
            
        # Plot data
        plt.imshow(chVol[sectionIndex,extentIndices[3]:extentIndices[2],extentIndices[0]:extentIndices[1]], cmap = self.colormaps[ch], 
                   vmin = 0, vmax = 600, extent = extent, alpha = 1, interpolation='none')
        if ticks:
            plt.title(ch)
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
        else:
            plt.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)
    
    def getCellsCCFdf(self, ch: list):
        """ 
        Retrieves and formats CCF transformed coordinates of segmented cells into a dataframe. 

        Parameters
        ----------
        ch : list of str 
            List of imaging channels to retrieve coordinates from (e.g., ["488", "561"]). 

        Returns
        -------
        location_df : pd.DataFrame
            Dataframe where each row is a cell and each column is a coordinate: AP (anterior-posterior), DV (dorsal-ventral), or ML (medial-lateral), with an additional "channel" column indicating the channel of origin. 
            
        """

        ccfDim = [528, 320, 456]
        locCells_list = []

        for channel in ch:
            locCells = pd.read_xml(self.ccfCellsPaths[channel], xpath="//CellCounter_Marker_File//Marker_Data//Marker_Type//Marker")
            locCells = locCells[['MarkerZ', 'MarkerY', 'MarkerX']]  # Rearrange indices to be AP, DV, ML
            locCells = locCells.rename(columns={'MarkerZ': 'AP', 'MarkerY': 'DV', 'MarkerX': 'ML'})  # Rename columns
            locCells = locCells.assign(channel=channel)  # Adds a column with the channel name

            # Clip coordinates to be within specified dimensions
            locCells['AP'] = locCells['AP'].clip(0, ccfDim[0]-1)
            locCells['DV'] = locCells['DV'].clip(0, ccfDim[1]-1)
            locCells['ML'] = locCells['ML'].clip(0, ccfDim[2]-1)

            locCells_list.append(locCells) 

        location_df = pd.concat(locCells_list, ignore_index=True)
        
        return location_df
    