from skimage import data, io, filters,transform
from scipy import misc
import os
import matplotlib.pyplot as plt
import numpy as np
import sys, getopt
from scipy.interpolate import interp1d
import cv2


def findShift(img,st=-9,en=10):
    pimg = np.max(img,axis=0)
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

    plt.figure()
    plt.plot(searchinterval, norms.T, 'r+',xp, f2(xp), 'g-')
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
    # plt.figure()
    # plt.plot(searchinterval, norms.T, 'r+',xp, f2(xp), 'g-')
    # return searchinterval[np.argmax(norms)]
    return int(np.round(shiftval)),shiftval

def main(argv):
    thumb = True
    inputfolder = None #
    inputfolder = "/groups/mousebrainmicro/mousebrainmicro/data/2017-10-31/Tiling/2017-11-02/01/01190"
    outputfolder = None
    saveout = False
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('linefix.py -i <inputfolder> -o <outputfolder>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('linefix.py -i <inputfolder> -o <outputfolder>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfolder = arg
        elif opt in ("-o", "--ofile"):
            outputfolder = arg

    if inputfolder==None:
        print('linefix.py -i <inputfolder> -o <outputfolder>')
        sys.exit(2)

    if outputfolder==None:
        outputfolder = inputfolder
        saveout = True
    results = [each for each in os.listdir(inputfolder) if each.endswith('.tif')]
    # read image
    imgori = io.imread(inputfolder+"/"+results[0])
    img = imgori/2**16
    # beta correction
    img = img** (1 / 2.2)
    dims = np.int64(np.shape(img))
    st = -9
    en = 10
    shift,shift_float = findShift(img,st,en)
    # check if shift is closer to halfway. 0.4<|shift-round(shift)|<0.6
    if np.abs(np.abs(np.round(shift_float,2)-np.round(shift_float,0))-.5)<.1:
        shift, shift_float = findShift3D(img,st,en)

    with open(outputfolder+'/Xlineshift.txt', 'w') as f:
        f.write('{0:d}'.format(shift))
        if thumb:
            cmap = plt.get_cmap('seismic',en-st)
            col = cmap(shift-st)
            thumbim = np.ones((105,89,3),dtype=np.uint8)
            col = tuple(c * 255 for c in col)
            thumbim[:] = col[:3]
            io.imsave(outputfolder + "/Thumbs.png", thumbim)

    #sizes = [(120, 120), (720, 720), (1600, 1600)]
    # for size in sizes:
    #     resized_image = cv2.resize(image, size)
    #     cv2.imwrite("thumbnail_%d.jpg" % size[0], resized_image)


    if saveout:
        # overwrite images
        for res in results:
            img = io.imread(inputfolder + "/" + res)
            img[:,1::2,:] =  np.roll(img[:,1::2,:], shift, axis=2)
            io.imsave(outputfolder+"/"+res,img)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
