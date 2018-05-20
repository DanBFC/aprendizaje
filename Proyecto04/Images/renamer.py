#!/usr/bin/python
import os, sys
from PIL import Image
from resizeimage import resizeimage

# This program renames and resizes images, it was made for resizing 1:1 aspect ratio images.
# but it can be easly modified with the "pythom-resize-image" library,
# it has some methods already implemented to do so like: "resize_thumbnail", which is the one in use in this program

# You also need to provide your own path to the images, provide it in the 'path' variable.

# Also when you save the image (im.save), you can also modify the name, this one names all of them "CurieX.jpg"
# 'X' being a number from 0 to n (n being the total amount of images).

# As a final note, this program dumps the newly resized images into the same directory of the python program,
# if you need to process a lot of images, consider cleaning you program's folder or placing it in a new folder.
# To prevent making a mess of your files of course.


#path = "/home/tredok/Documents/RPAA/aprendizaje/Proyecto02/DataSet/Testing/"

path = "/home/tredok/Documents/aprendizaje/Proyecto04/catImages/curie/"  # Path to the images 
images = os.listdir(path)                   # Read the files inside the provided path
i = 339                                     # Start a counter in 0

for image in images:

    if(image.startswith("re")):  # this was just to check there were no strange files
       continue                  #(i had some that started with "re" and wanted to ignore them
   
    if os.path.isfile(path+image):
        print "resizing: " + image
        im = Image.open(path+image)
        im = resizeimage.resize_thumbnail(im, [150, 150])
        im.save('Curie' + str(i) + '.jpg', im.format)
        im.close()
        print image + " resized"
    i += 1
