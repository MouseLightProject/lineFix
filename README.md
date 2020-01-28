# linefix
Checks a 3D tif stack has line shift artifact using even/odd line correlation. It runs a greedy search for shifted even lines. 
## How to run:
linefix.py -i <inputfile> -o <outputfile>
inputfolder: folder that has tif file(s)
output folder: [optional], if provided, tif files in input folder will be corrected based on the shift of the first folder and exported into this folder 
