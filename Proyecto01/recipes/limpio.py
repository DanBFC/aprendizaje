import json

def limpia_cadena(cadena):
    cadena=cadena.encode('ascii', 'ignore').decode('ascii')
    cadena=cadena.replace('"','')
    cadena=cadena.replace("'",'')
    cadena=cadena.replace(",",' ')
    cadena=cadena.replace(".",'')
    return cadena
#print data[0]

json_data=open("recipes.json").read()
data=json.loads(json_data)

archivo=open('recetas.arff','w')
for j in data:
    linea=''
    entrada1= limpia_cadena( j['directions'])
    entrada2= j['name']
    aux=''
    for i in j['ingredients']:
        aux+=limpia_cadena(i)+' '
        
    
    archivo.write("'"+entrada1+"','"+entrada2+"','"+aux+"' \n")
    print '\n'
#for i in data:
    #print i
