import os, sys
from PIL import Image

path_to_cat_images = "/home/tredok/Documents/aprendizaje/Proyecto02/Curie/"
path_to_non-cat_images = "/home/tredok/Documents/aprendizaje/Proyecto02/nonCurie/"


def curie_image_reader(path_to_curie, label):
    curieArray = []
    curieImages = os.listdir(path_to_curie)

    for image in curieImages:
        curie = open(image, 'r+')
        curieArray.append([curie, label])
    return currieArray
