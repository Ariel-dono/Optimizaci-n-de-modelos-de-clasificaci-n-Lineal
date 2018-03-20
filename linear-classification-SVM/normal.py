# Imports
from data_loading import *
from scipy.stats import chi2
from scipy.stats import norm
import random

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
    return (w)


# randomly generate the first generation
def set_up_populationNorm(probabilityDistribution, individualSize, populationSize, scaleGeneration, bins):
    population = np.array([np.array(generateW(bins,individualSize), dtype=float)])
    for i in range(1, populationSize):
        population = np.append(population, [np.array(generateW(bins,individualSize), dtype=float)], axis=0)
    return population

# randomly generate the first generation
def set_up_population(probabilityDistribution, individualSize, populationSize, scaleGeneration, bins):
    population = np.array([np.array(probabilityDistribution(int(scaleGeneration/2), size=[bins,individualSize]), dtype=int)])
    for i in range(1, populationSize):
        population = np.append(population, [np.array(probabilityDistribution(int(scaleGeneration/2), size=[bins, individualSize]), dtype=int)], axis=0)
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
    while(minIndex!=(len(wMatrix)//2)): #stablish the middle 
        pairMatches.append([wMatrix[minIndex],wMatrix[maxIndex-1]])
        minIndex+=1
        maxIndex-=1
    return (pairMatches)

# fittest individuals selection
def fittest_selection_matching(fitest_individuals):
    selector = 0
    result = []
    while selector+1 < len(fitest_individuals):
        selected = fitest_individuals[selector]
        mates = fitest_individuals[selector+1:]
        for element in mates:
            result = result + [[selected, element]]
        selector += 1
    return result


#-----------------------------------------------------------#
# name: mutationAlgthm
# description: function that determine which chromosome 
#-----------------------------------------------------------#
def mutationAlgthm(poblation):
    mu, sigma = 2, 0.5
    mutationNum = np.random.normal(mu,sigma)
    randOper= list(np.random.randn(4))
    randGen= list(np.random.randn(len(poblation[0])))
    randClass= list(np.random.randn(len(poblation)))
    _Class=randClass.index(max(randClass))
    _Gen=randGen.index(max(randGen))
    _Oper=randOper.index(max(randOper))

    if _Oper == 0:
        #print("add")
        value=poblation[_Class][_Gen]+ mutationNum
        poblation[_Class][_Gen]=value
   
    elif _Oper == 1:
        #print("mult")
        value=poblation[_Class][_Gen]* mutationNum
        poblation[_Class][_Gen]=value
        
    elif _Oper == 2:
        #print("minus")
        value=poblation[_Class][_Gen]- mutationNum
        poblation[_Class][_Gen]=value
        
    elif _Oper == 3:
        #print("div")
        value=poblation[_Class][_Gen]/ mutationNum
        poblation[_Class][_Gen]=value
        
    else:
        # Do the default
        print ("There's something wrong!!")
                
    return poblation
# chromosomic mutations
def mutate_population(child):
    result = child
    mutations_index = np.array(np.abs(norm.rvs(scale=len(child)/2, size=2)), dtype=int)
    resetInvalidPositions = np.array(mutations_index < len(child), dtype=int)
    mutations_index = mutations_index * resetInvalidPositions
    mutableData = result[mutations_index[0]]
    result[mutations_index[0]] = mutableData + random.randrange(22)
    mutableData = result[mutations_index[1]]
    result[mutations_index[1]] = mutableData - random.randrange(22)
    return result


def crossoverNorm(pMutation, pParentA, pParentB, pDominant):
    if (pDominant==1):
        newIndvidual = pParentA
    else:
        newIndvidual = pParentB
    return (newIndvidual)
def crossover(mutate, parentA, parentB, dominance):
    if dominance:
        result = parentA
        boundedTransfer = int((len(result) * 10) / 100);
        result[:boundedTransfer] = parentB[:boundedTransfer]
    else:
        result = parentB
        boundedTransfer = int((len(result) * 30) / 100);
        result[:boundedTransfer] = parentA[:boundedTransfer]
    return mutate(result)
def getDominant():
    
    s = np.random.randn()
    if (int (np.absolute(s))>2):
        return int (np.absolute(np.random.random_sample(1)))
    return int (np.absolute(s))

def dominant_selectionNorm(pMutation, pCrossover, pPair):
    classes = len(pPair[0])
    newIndvidual = np.zeros((classes, len(pPair[0][0])))
    for i in range(0, classes):
        dominant = getDominant()
        newIndvidual[i] = pCrossover(pMutation, pPair[0][i], pPair[1][i], dominant)
    mutate=getDominant()
    if mutate:
        return pMutation(newIndvidual)
    else:
        return(newIndvidual)


# dominant chromosomic selection
def crossingNorm(pMutation, pCrossover, wPairs):
    newGen = []
    for pair in wPairs:
        newGen = newGen + [dominant_selectionNorm(pMutation, pCrossover, pair)]
    return newGen
def dominant_selection(mutate, crossover, pair):
    classes = len(pair[0])
    result = np.zeros((classes, len(pair[0][0])))
    dominant = True
    for i in range(0, classes):
        result[i] = crossover(mutate, pair[0][i], pair[1][i], dominant)
        dominant = not dominant
    return result


# dominant chromosomic selection
def crossing(mutate, crossover, pairs):
    result = []
    for pair in pairs:
        result = result + [dominant_selection(mutate, crossover, pair)]
    return result

# adaptability fittest function: (SVM powered linear classifier)
def linear_classifier(svmPopulation, data, quantity):
    population_loss = np.array([])
    for svm in svmPopulation:
        global_loss = 0
        quantitySamples = len(data[0])
        for i in range(0, quantitySamples):
            classification = np.dot(svm, data[0][i])
            tag = int(data[1][i])
            classificationDiff = (classification - classification[tag]) + 1
            classificationDiff = np.append(classificationDiff[:tag], classificationDiff[tag + 1:])
            argHingeMax = np.array(classificationDiff > 0, dtype=int)
            global_loss = global_loss + np.sum(classificationDiff * argHingeMax)
        population_loss = np.append(population_loss, [global_loss/quantitySamples])
        sortedByArgs = np.argsort(population_loss)
    return sortedByArgs[:quantity], population_loss[sortedByArgs[0]]


# data loading functions
def cifar10_loader():
    return loadCIFAR10()


def iris_loader():
    print("loading Iris")


# Getting acceptable solution
def stop_condition(genValidation, loss, threshold):
    if threshold >= round(loss[0]):
        entering = False
    else:
        if len(loss) >= genValidation:
            lastGeneration = loss[0]
            lastestGenerations = loss[:genValidation]
            invariabilityIndex = np.sum(np.array(lastestGenerations == lastGeneration, dtype=int))
            if int(invariabilityIndex) == genValidation:
                entering = False
            else:
                entering = True
        else:
            return True
    return entering


# data visualization
def data_visualization(experimentsResults):
    print(experimentsResults)
    print("Showing results")


def loss_index(adaptability, lossLog, population, data, quantity):
    adaptabilitySelectionIndex, generationBetterLoss = adaptability(population, data, quantity)
    result = np.array([population[adaptabilitySelectionIndex[0]]])
    for i in range(1, len(adaptabilitySelectionIndex)):
        result = np.append(result, np.array([population[adaptabilitySelectionIndex[i]]]), axis=0)
    return result, np.append([generationBetterLoss], lossLog)


# linear classification model
def linear_model_optimization(probabilityDistribution, initializer, matching, mutate, crossing, crossover,
                              adaptability, loadData, genValidation, correctnessAnalysis,
                              optimalIndividual, threshold, nValidationGeneration, populationSize, scaleGeneration, bins):
    lossLog = []
    data = loadData()
    population = initializer(probabilityDistribution, len(data[0][0]), populationSize, scaleGeneration, bins)
    OptimalSVMLog, lossLog = loss_index(adaptability, lossLog, population, data, optimalIndividual)
    generation = 1
    while genValidation(nValidationGeneration, lossLog, threshold):
        print("Generation:", generation, "Best loss:", lossLog[0], "Tamaño:", len(population))
        population = crossing(mutate, crossover, matching(OptimalSVMLog))
        population = np.append(population, OptimalSVMLog, axis=0)
        OptimalSVMLog, lossLog = loss_index(adaptability, lossLog, population, data, optimalIndividual)
        generation += 1
    correctnessAnalysis(lossLog)


##linear_model_optimization(chi2.rvs, set_up_population, fittest_selection_matching, mutate_population,
##                          crossing, crossover, linear_classifier, loadCIFAR10, stop_condition,
##                          data_visualization, 16, 80, 4, 160, 256, 4)


#---- Normal-----#
linear_model_optimization(norm.rvs, set_up_populationNorm, matching, mutationAlgthm,
                          crossingNorm, crossoverNorm, linear_classifier, loadCIFAR10, stop_condition,
                          data_visualization, 16, 0, 4, 160,256, 4)
