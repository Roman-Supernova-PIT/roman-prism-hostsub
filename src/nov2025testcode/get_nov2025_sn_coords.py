from snappl.diaobject import DiaObject
from snappl.image import Image

import os
import sys
import numpy as np
from astropy.io import fits

# import pdb

# Each diaobject ID corresponds to a transient
# This is the diaobject ID for the reference SN
diaobject_id = 'b1fe9225-3784-4e7d-b68d-121d8c29f443'
dobj = DiaObject.get_object(diaobject_id=diaobject_id)
# This SN appears in all the prism images
# so len(dispimages) are all the prism images available.
dispimages = Image.find_images(provenance_tag='ou2024',
                               process='load_ou2024_dispimage',
                               ra=dobj.ra, dec=dobj.dec)
print('Total prism images:', len(dispimages))

# ===============
"""
# Find other transients within some radius [arcsec]
objs = DiaObject.find_objects(provenance=dobj.provenance_id,
                              ra=dobj.ra, dec=dobj.dec,
                              radius=2000)

print('Other transient objs within 2000 arcsec of reference SN:',
      len(objs))
# Given a large search radius, like 2000 arcsec, you will
# find all the transient objects within the database.
# pdb.set_trace()
# sys.exit(0)

mjds = [i.mjd for i in dispimages]
minmjd = min(mjds)
maxmjd = max(mjds)

with open('nov2025_transients_prismimg_matches.txt', 'w') as fh:
    fh.write('# objra objdec objpeakmjd prismimgname '
             + 'prismimgmjd prismimgexptime')
    fh.write('\n')

    for o in objs:
        mjd_peak = o.mjd_peak
        if mjd_peak >= minmjd and mjd_peak <= maxmjd:

            # Now find the prism images where this object appears.
            # Note that each one of these transients has some LC
            # according to which it was simulated and will
            # appear in multiple prism images.
            prismimgs = Image.find_images(provenance_tag='ou2024',
                                          process='load_ou2024_dispimage',
                                          ra=o.ra, dec=o.dec)
            if len(prismimgs) == 0:
                continue
            print('\n', o.ra, o.dec, mjd_peak)
            sortindex = np.argsort([img.mjd for img in prismimgs])
            mjds = np.array([img.mjd for img in prismimgs])[sortindex]
            path = np.array([os.path.basename(img.path)
                             for img in prismimgs])[sortindex]

            # Find prism image closest to object's peak MJD
            closepeak_idx = np.argmin(abs(mjds - mjd_peak))
            print(path[closepeak_idx], mjds[closepeak_idx])

            hdr = fits.getheader(, extname='SCI')
            exptime = hdr['EXPTIME']

            # Write to file
            fh.write('{:.8f}'.format(o.ra) + ' ')
            fh.write('{:.8f}'.format(o.dec) + ' ')
            fh.write('{:.4f}'.format(mjd_peak) + ' ')
            fh.write(path[closepeak_idx] + ' ')
            fh.write('{:.4f}'.format(mjds[closepeak_idx]))
            fh.write('{:.2f}'.format(exptime))
            fh.write('\n')

sys.exit()
"""

# Create a dict of all direct images
alldirimagesdict = {}

rband_imgs = Image.find_images(provenance_tag='ou2024',
                               process='load_ou2024_image',
                               ra=dobj.ra, dec=dobj.dec, band='R062')
zband_imgs = Image.find_images(provenance_tag='ou2024',
                               process='load_ou2024_image',
                               ra=dobj.ra, dec=dobj.dec, band='Z087')
yband_imgs = Image.find_images(provenance_tag='ou2024',
                               process='load_ou2024_image',
                               ra=dobj.ra, dec=dobj.dec, band='Y106')
jband_imgs = Image.find_images(provenance_tag='ou2024',
                               process='load_ou2024_image',
                               ra=dobj.ra, dec=dobj.dec, band='J129')
hband_imgs = Image.find_images(provenance_tag='ou2024',
                               process='load_ou2024_image',
                               ra=dobj.ra, dec=dobj.dec, band='H158')
fband_imgs = Image.find_images(provenance_tag='ou2024',
                               process='load_ou2024_image',
                               ra=dobj.ra, dec=dobj.dec, band='F184')

alldirimagesdict['R062'] = rband_imgs
alldirimagesdict['Z087'] = zband_imgs
alldirimagesdict['Y106'] = yband_imgs
alldirimagesdict['J129'] = jband_imgs
alldirimagesdict['H158'] = hband_imgs
alldirimagesdict['F184'] = hband_imgs


def find_dir_match(pointing, sca, band=None):

    dirimages = alldirimagesdict[band]

    dirmatch_flag = 0
    for j in range(len(dirimages)):
        img = dirimages[j]
        dirbpath = os.path.basename(img.path)
        dirbpath = dirbpath.replace('.fits.gz', '')
        dirsplit = dirbpath.split('_')
        dirpointing = dirsplit[5]
        dirsca = dirsplit[6]

        if (dirpointing == pointing) and (dirsca == sca):
            dirmatch_flag = 1
            break

    return dirbpath, dirmatch_flag


# We want a matched list of prism and direct images
# We need to match the pointing and SCA. This needs to
# be the same for the dispersed and direct image.
dispmatchedlist = []
dirmatchedlist = []

for i in range(len(dispimages)):
    currentimg = dispimages[i]
    cpath = currentimg.path
    bpath = os.path.basename(cpath)
    bpath = bpath.replace('.fits.gz', '')
    bsplit = bpath.split('_')
    pointing = bsplit[5]
    sca = bsplit[6]

    print('\n', i, bpath, pointing, sca)

    # Now find this pointing and sca in the direct images
    # First search the J-band images and then if no match
    # is found search H-band
    allbands = ['R062', 'Z087', 'Y106', 'J129', 'H158', 'F184']
    dirmatch_flag = 0
    count = 0
    while dirmatch_flag == 0:
        bandname = allbands[count]
        print('\nChecking match for:', bandname)
        dirbpath, dirmatch_flag = find_dir_match(pointing, sca, band=bandname)
        print(dirbpath, dirmatch_flag)
        count += 1
        if count >= 5:
            break

    # Append to list
    if dirmatch_flag:
        dispmatchedlist.append(bpath)
        dirmatchedlist.append(dirbpath)

# Write to txt file
with open('nov2025_disp_dir_matched_images.txt', 'w') as fh:
    fh.write('# DispImgName DirImgName')
    fh.write('\n')
    for k in range(len(dispmatchedlist)):
        dispimg = dispmatchedlist[k]
        dirimg = dirmatchedlist[k]
        fh.write(dispimg + ' ')
        fh.write(dirimg + ' ')
        fh.write('\n')

sys.exit(0)
