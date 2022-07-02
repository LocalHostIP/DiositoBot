from torch import classes

def cargarClases(path):
    clases=[]
    f = open(path,'r')
    for c in f:
        clases.append(c.replace('\n',''))
    f.close()
    return clases
    
print(cargarClases("data/coco.names"))