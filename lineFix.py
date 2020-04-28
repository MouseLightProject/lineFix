import os
import sys
import getopt
import numpy as np
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import re



def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]



def findShift(img,st=-9,en=10,isdeployed=False):
    pimg = np.max(img,axis=0)
    # if False:
    #     im1 = np.asarray(np.tanh(pimg[::2])>.5,np.float)
    #     im2 = np.asarray(np.tanh(pimg[1::2])>.5,np.float)
    # else:
    im1 = pimg[::2]
    im2 = pimg[1::2]

    norms=np.zeros((1,en-st))
    searchinterval = range(st,en)

    for iter,shift in enumerate(searchinterval):
        corr = im1*np.roll(im2,shift,axis=1)#/np.linalg.norm(im1)/np.linalg.norm(im2)
        norms[0,iter] = np.linalg.norm(corr)/np.linalg.norm(im1)/np.linalg.norm(im2)

    xp = np.linspace(st,en-1, num=1000, endpoint=True)
    f2 = interp1d(searchinterval, norms.flatten(), kind='cubic')
    shiftval = xp[np.argmax(f2(xp))]

    if not isdeployed:
        plt.figure()
        plt.imshow(im1)
        plt.figure()
        plt.plot(searchinterval, norms.T, 'r+',xp, f2(xp), 'g-')
        plt.title(shiftval)

    # return searchinterval[np.argmax(norms)]
    return int(np.round(shiftval)),shiftval



def sliceByFix(img):
    corrslices=np.zeros((img.shape[0],5))
    for iter,slice in enumerate(img):
        im1 = slice[::2]
        im2 = slice[1::2]
        IM1 = transform.resize(im1,np.array(im1.shape)*np.array([2,1]), mode='constant')
        IM2 = transform.resize(im2,np.array(im2.shape)*np.array([2,1]), mode='constant')

        IM1 = IM1[1:]
        IM2 = IM2[0:-1]

        rl=-1
        IM = np.zeros((IM1.shape[0], IM1.shape[1], 3))
        IM[:, :, 0] = IM1
        IM[:, :, 2] = IM2 * 0

        IM[:, :, 1] = np.roll(IM2,rl,axis=1)
        plt.figure(),
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(IM), plt.title('sh: {0}'.format(rl))

        rl=0
        IM[:, :, 1] = np.roll(IM2,rl,axis=1)
        ax2 = plt.subplot(1, 3, 2, sharex=ax1)
        ax2.imshow(IM), plt.title('sh: {0}'.format(rl))

        rl=1
        IM[:, :, 1] = np.roll(IM2,rl,axis=1)
        ax3 = plt.subplot(1, 3, 3, sharex=ax1)
        ax3.imshow(IM), plt.title('sh: {0}'.format(rl))


        for rl in range(-2,3):
            corr = np.linalg.norm(IM1[2::-1]*np.roll(IM2[2::-1],rl,axis=1))
            corrslices[iter,rl+2] = corr
    return corrslices



def findShift3D(img,st=-10,en=10):
    im1 = img[:,::2,:]
    im2 = img[:,1::2,:]
    if im1.shape[1]>im2.shape[1]:
        im1 = np.delete(im1,im1.shape[1]-1,1)
    norms=np.zeros((1,en-st))
    searchinterval = range(st,en)
    for iter,shift in enumerate(searchinterval):
        corr = im1*np.roll(im2,shift,axis=2)#/np.linalg.norm(im1)/np.linalg.norm(im2)
        norms[0,iter] = np.linalg.norm(corr)/np.linalg.norm(im1)/np.linalg.norm(im1)

    xp = np.linspace(st,en-1, num=1000, endpoint=True)
    f2 = interp1d(searchinterval, norms.flatten(), kind='cubic')
    shiftval = xp[np.argmax(f2(xp))]
    return int(np.round(shiftval)),shiftval



def neurons_channel_index_from_file_lines(file_lines):
    for file_line in file_lines:
        tokens = file_line.split(':')
        if len(tokens) >= 2 :
            channel_index_as_string = tokens[0].strip()
            channel_semantics_as_string = tokens[1].strip()
            if channel_semantics_as_string == 'neurons':
                result = int(channel_index_as_string)
                return result
    raise RuntimeError('Unable to determine the index of the neurons channel')



def main(argv):
    thumb = True
    isdeployed = True
    input_root_folder = None
    output_root_folder = None
    tile_relative_path = None
    do_write_output_tif_stacks = False

    if isdeployed:
        do_write_output_tif_stacks = True

    try:
        opts, args = getopt.getopt(argv, "hi:o:p:", ["ifile=", "ofile=", "path="])
    except getopt.GetoptError:
        print('lineFix.py -i <input_root_folder> -p <tile path> -o <output_root_folder>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('lineFix.py -i <input_root_folder> -p <tile path> -o <output_root_folder>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_root_folder = arg
        elif opt in ("-o", "--ofile"):
            output_root_folder = arg
        elif opt in ("-p", "--path"):
            tile_relative_path = arg

    if input_root_folder==None:
        print('lineFix.py -i <input_root_folder> -p <tile path> -o <output_root_folder>')
        sys.exit(2)

    if tile_relative_path==None:
        print('lineFix.py -i <input_root_folder> -p <tile path> -o <output_root_folder>')
        sys.exit(2)

    if output_root_folder==None:
        print('lineFix.py -i <input_root_folder> -p <tile path> -o <output_root_folder>')
        sys.exit(2)

    # Figure out which channel we'll use to compute the line shift
    channel_semantics_file_path = os.path.join(input_root_folder, 'channel-semantics.txt')
    with open(channel_semantics_file_path) as f:
        channel_semantics_lines = list(f)
    neurons_channel_index = neurons_channel_index_from_file_lines(channel_semantics_lines)

    # Construct absolute paths to input, output folder for this tile
    input_folder_path = os.path.join(input_root_folder, tile_relative_path)
    output_folder_path = os.path.join(output_root_folder, tile_relative_path)
    tile_base_name = os.path.basename(tile_relative_path)

    # # Get the list of tif files in the input folder
    # unsorted_tif_file_names = [file_name for file_name in os.listdir(input_folder_path) if file_name.endswith('.tif')]
    # tif_file_names = sorted(unsorted_tif_file_names, key=natural_sort_key)

    # read image
    #neurons_channel_tif_file_name = tif_file_names[neurons_channel_index]
    neurons_channel_tif_file_name = tile_base_name + '-ngc.' + str(neurons_channel_index) + '.tif'
    neurons_channel_tif_file_path = os.path.join(input_folder_path, neurons_channel_tif_file_name)
    imgori = io.imread(neurons_channel_tif_file_path)
    img = imgori/2**16
	
    # gamma correction
    img = img** (1 / 2.2)
	
    # binarize it to eliminate spatial non-uniformity bias
    img = np.asarray(np.tanh(img[::2])>.5,np.float)
    st = -9
    en = 10
    shift,shift_float = findShift(img,st,en,isdeployed)
    # check if shift is closer to halfway. 0.4<|shift-round(shift)|<0.6
    if np.abs(np.abs(np.round(shift_float,2)-np.round(shift_float,0))-.5)<.1:
        shift, shift_float = findShift3D(img,st,en)

    # Make sure the output folder exists
    os.makedirs(output_folder_path, exist_ok=True)

    # Write the Xlineshift.txt file
    xlineshift_file_path = os.path.join(output_folder_path, 'Xlineshift.txt')
    with open(xlineshift_file_path, 'w') as f:
        f.write('{0:d}'.format(shift))

    # Write the thumbnail, maybe
    if thumb:
        cmap = plt.get_cmap('seismic',en-st)
        col = cmap(shift-st)
        thumbim = np.ones((105,89,3),dtype=np.uint8)
        col = tuple(c * 255 for c in col)
        thumbim[:] = col[:3]
        thumb_file_path = os.path.join(output_folder_path, 'Thumbs.png')
        io.imsave(thumb_file_path, thumbim)

    # Write the line-shifted tif stacks, maybe
    tif_file_names = [file_name for file_name in os.listdir(input_folder_path) if file_name.endswith('.tif')]
    if do_write_output_tif_stacks:
        for tif_file_name in tif_file_names:
            input_tif_file_path = os.path.join(input_folder_path, tif_file_name)
            output_tif_file_path = os.path.join(output_folder_path, tif_file_name)
            img = io.imread(input_tif_file_path)
            img[:,1::2,:] =  np.roll(img[:,1::2,:], shift, axis=2)
            io.imsave(output_tif_file_path, img)



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
