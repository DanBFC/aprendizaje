def reader():
    recipes = "recetas.arff"
    recipesAr = []
    with open(recipes, 'r') as f:
        for line in f:
            #recipe = f.readline()
            splitedRecipe = line.split(",")
            recipesAr.append(splitedRecipe)
    print(recipesAr)
    return recipesAr

def writer_of_files(recipes):
    #print(recipes)
    for recipe in recipes:
        #print(recipe[2])
        recipeName = recipe[1][1 : -1] + ".txt"
        recipePrep = recipe[0]
        recipeIng = recipe[2]
        file = open(recipeName, "w")
        file.write(recipeIng)
        file.write(recipePrep)

if __name__ == "__main__":
    recetas = reader()
    writer_of_files(recetas)
