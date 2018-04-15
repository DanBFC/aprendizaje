import os, sys




def image_reader(path_to_curie, label):
    curieArray = []
    curieImages = os.listdir(path_to_curie)

    # for image in curieImages:
    #     #print image
    #     curie = open(path_to_curie + image, 'r+')
    #     curieArray.append([curie, label])
    # return curieArray


# if __name__ == "__main__":
#     path_to_curie = "/home/tredok/Documents/aprendizaje/Proyecto02/Curie/"
#     path_to_nonCurie = "/home/tredok/Documents/aprendizaje/Proyecto02/nonCurie/"

#     curie_DataSet = image_reader(path_to_curie, 1)
#     nonCurie_DataSet = image_reader(path_to_nonCurie, 0)
#     print nonCurie_DataSet + curie_DataSet 
#     print curie_DataSet
