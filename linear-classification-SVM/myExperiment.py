# imports
import numpy as np
import cPickle
import math
import operator
import time;  # This is required to include time module.
import matplotlib.pyplot as plt
from matplotlib import cm
from sys import version_info
np.seterr(over='ignore')

#-----------------------------------------------------------#
# name: loadDataIris
# description:
#-----------------------------------------------------------#





#-----------------------------------------------------------#
#              Section name: loadDataCIFAR                  #
#-----------------------------------------------------------#
def unpickle(file):
    # Load the dictionary of images for this batch
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def loadCifarGray(pathname):

    batch=unpickle(pathname)
   
    # The 'data' part of the dicitionary contains the actual image pixels.  
    imgdata = batch['data']

    # Combine the R, G, and B components together to get grayscale, using the luminosity-preserving formula:

    grayscale = 0.21*imgdata[:,0:1024] + 0.72*imgdata[:,1024:2048] + 0.07*imgdata[:,2048:3072]

    return grayscale, np.array(batch['labels']), batch['filenames']
def chargeImgs(dataFile,namesFile):
    myImages, labels_index, names = loadCifarGray(dataFile)
    myTags=unpickle(namesFile)
    imagesTags=[]
    fullMatrix=[]
    for tagIndex in labels_index:
        imagesTags.append(myTags['label_names'][tagIndex])
    for i in range(len(myImages)):
        fullMatrix.append((myImages[i],imagesTags[i]))
   # print repr(np.array(fullMatrix)[0:3])
    return np.array(fullMatrix)


def displayImgs(dataFile,nameFile,cantImgs):
    images, labels_index, names = loadCifarGray(dataFile)

    diccionary=unpickle(nameFile)
    labels=diccionary['label_names']
    # Display the image, with its name and label in one window you should close it to watch the next one
    for index in range(cantImgs):
        image = images[index]
        plt.imshow(image.reshape(32,32), cmap=cm.gray)
        plt.title('name='+names[index] + '  label=' + labels[(labels_index[index])])
        plt.show()
       

#-----------------------------------------------------------#
# name: normalDistribution
# description: normal distribution function that will help
#              to determine how we get our first Poblation
#-----------------------------------------------------------#
def generateW(pCantClasses,pImgDimention):
    mu, sigma = 1.85, 0.5
    bias = np.random.normal(mu,sigma,pCantClasses)
    normalArray=np.random.normal(mu,sigma,pImgDimention*pCantClasses)
    
##    count, bins, ignored = plt.hist(normalArray, 30, normed=True)
##    plt.show()
    w=[]
    for i in range(pCantClasses) :
        Wi=[]
        cont=0
        for j in range(pImgDimention):
            cont=j
            Wi.append(normalArray[j])
            
        normalArray=normalArray[cont+1:]
        
        w.append(Wi)
    i =0
    while (i!=(len(bias))):
        for each in w:
            each.append(bias[i])
            i+=1
    #print np.array(w)
    return (np.array(w))
from scipy.stats import chi2
from scipy.stats import norm
import random


# randomly generate the first generation
def set_up_population(probabilityDistribution, individualSize, populationSize, scaleGeneration, bins):
    population = np.array([np.array(probabilityDistribution(int(scaleGeneration/2), size=[bins, individualSize]), dtype=int)])
    for i in range(1, populationSize):
        population = np.append(population, [np.array(probabilityDistribution(int(scaleGeneration/2), size=[bins, individualSize]), dtype=int)], axis=0)
    print len(population[0][0])   
    return population

#-----------------------------------------------------------#
# name: matching
# description: function that matches the individuals of the
#              population. Matchs the first individual with
#              the last.
#-----------------------------------------------------------#
def matching(wMatrix):
    minIndex=0
    maxIndex=len(wMatrix)
    pairMatches=[]
    while(minIndex!=(len(wMatrix)//2)):
        
        pairMatches.append((wMatrix[minIndex],wMatrix[maxIndex-1]))
        minIndex+=1
        maxIndex-=1
    return np.array(pairMatches)
print set_up_population(norm.rvs,1025,6,256,4)
print (matching(set_up_population(norm.rvs,1025,6,256,4)))        
#-----------------------------------------------------------#
# name: dominantSelection
# description: function that determines which in each pair
#              is the dominant. Generate the new poblation.
#-----------------------------------------------------------#
def getDominant():
    
    s = np.random.randn()
    if (int (np.absolute(s))>2):
        return int (np.absolute(np.random.random_sample(1)))
    return int (np.absolute(s))
def domainSelection(wPairs):
    percentage=30
    return domainSelection_Aux(wPairs,percentage)
def domainSelection_Aux(wPairs,percentage):
    index=(len(wPairs))
    newGen=[]
    for pair in wPairs:
##        print pair
        dominant=getDominant()
##        print dominant
        value=[]
        j=0
        if (dominant==1):
            for i in pair[0]:
                #print (percentage*i/100) + pair[1][j]
                value.append((percentage*i/100)+ pair[1][j])
                j+=1
            newGen.append(value)
                
            
        else:
            for i in pair[1]:
##                print (percentage*i/100) + pair[1][j]
                value.append((percentage*i/100)+pair[0][j])
                             
                j+=1
            newGen.append(value)
##    print "newGen" + repr(np.array(newGen))   
    return np.array(newGen)
        
        
            
        
    


#-----------------------------------------------------------#
# name: mutationAlgthm
# description: function that determine which chromosome 
#-----------------------------------------------------------#

def mutationAlgthm(poblation):
    mu, sigma = 2, 0.5
    mutationNum = np.random.normal(mu,sigma)
    print mutationNum
    randOper= list(np.random.randn(4))
    randGen= list(np.random.randn(len(poblation[0])))
    randClass= list(np.random.randn(len(poblation)))
    print randOper
    print randGen
    print randClass
    print poblation
    _Class=randClass.index(max(randClass))
    _Gen=randGen.index(max(randGen))
    _Oper=randOper.index(max(randOper))

    if _Oper == 0:
        print("add")
        value=poblation[_Class][_Gen]+ mutationNum
        poblation[_Class][_Gen]=value
   
    elif _Oper == 1:
        print("mult")
        value=poblation[_Class][_Gen]* mutationNum
        poblation[_Class][_Gen]=value
        
    elif _Oper == 2:
        print("minus")
        value=poblation[_Class][_Gen]- mutationNum
        poblation[_Class][_Gen]=value
        
    elif _Oper == 3:
        print("div")
        value=poblation[_Class][_Gen]/ mutationNum
        poblation[_Class][_Gen]=value
        
    else:   
        # Do the default
        print ("There's something wrong!!")
                
    return poblation               
                 
     

     
     
    


#-----------------------------------------------------------#
# name: 
# description:
#-----------------------------------------------------------#

#TrainMatrix=(chargeImgs("data/data_batch_1","data/batches.meta"))
##TestMatrix=(chargeImgs("data/test_batch","data/batches.meta"))
##print TrainMatrix[:3]
#print len(TrainMatrix[0][0])
##
##displayImgs('data/data_batch_1',"data/batches.meta",2)
##print generateW(4,1024)
##print repr(matching(generateW(4,1024)))
#print repr(domainSelection(matching(generateW(4,1024))))
a= domainSelection(matching(generateW(4,1)))

print mutationAlgthm(a)
