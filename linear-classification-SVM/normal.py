# Imports
from data_loading import *
from scipy.stats import chi2
from scipy.stats import norm
from matplotlib import cm
import matplotlib.pyplot as plt
import random


def generateW(pCantClasses,pImgDimention):
    mu, sigma = 127, 75
    bias = np.random.normal(mu,sigma,pCantClasses)
    normalArray=np.abs(np.random.normal(mu,sigma,pImgDimention*pCantClasses))
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
    mu, sigma = 25, 10

    randOper= list(np.random.randn(4))

    randClass= list(np.random.randn(len(poblation)))
    _Class=randClass.index(max(randClass))
    _Oper=randOper.index(max(randOper))

    for i in range(0, len(poblation)):
        randGen = list(np.random.randn(len(poblation[0])))
        mutationNum = np.abs(np.random.normal(mu, sigma))
        mutationNum1 = np.abs(np.random.normal(mu, sigma))
        _Gen1 = randGen.index(max(randGen))
        #print("add")
        value=poblation[i][_Gen1]+ mutationNum1
        poblation[i][_Gen1]=value

        _Gen3 = randGen.index(max(randGen))
        #print("minus")
        value=poblation[i][_Gen3]- mutationNum
        poblation[i][_Gen3]=value
        _Gen4 = randGen.index(max(randGen))
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
    #mutate=getDominant()
    #if mutate:
    return pMutation(newIndvidual)
    #else:
    #    return(newIndvidual)


# dominant chromosomic selection
def crossingNorm(pMutation, pCrossover, wPairs):
    newGen = []
    for pair in wPairs:
        newGen = newGen + [dominant_selection(pMutation, pCrossover, pair)]
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

def hinge(global_loss, classification, tag):
    classificationDiff = (classification - classification[tag])
    classificationDiff = np.append(classificationDiff[:tag], classificationDiff[tag + 1:])
    argHingeMax = np.array(classificationDiff > 0, dtype=int)
    return global_loss + np.sum(classificationDiff * argHingeMax)


def softmax(global_loss, classification, tag):
    groundThruth = np.exp(classification[tag])
    classificationDenom = np.sum(np.exp(classification[:]))
    softmaxClassification = groundThruth/classificationDenom
    return global_loss + softmaxClassification


def cross_entropy(global_loss, classification, tag):
    return global_loss - np.log(softmax(global_loss, classification, tag))


# adaptability fittest function: (SVM powered linear classifier)
def linear_classifier(svmPopulation, data, quantity):
    population_loss = np.array([])
    for svm in svmPopulation:
        global_loss = 0
        quantitySamples = len(data[0])
        for i in range(0, quantitySamples):
            classification = np.dot(svm, data[0][i])
            tag = data[1][i]
            global_loss = hinge(global_loss, classification, tag)
        population_loss = np.append(population_loss, [global_loss/quantitySamples])
        sortedByArgs = np.argsort(population_loss)
    return sortedByArgs[:quantity], population_loss[sortedByArgs[0]]


# data loading functions
def load_cifar10():
    # Prefix path to datapools
    # All data pools must be in the same directory
    prefix = "/home/ariel/Documents/Inteligencia Artificial/cifar-10-python"
    return loadCIFAR10(prefix)


def load_correctness_data(testsQuantity):
    prefix = "/home/ariel/Documents/Inteligencia Artificial/cifar-10-python"
    test_data, labels = load_test_data(prefix)
    imgs = test_data[:testsQuantity]
    imgsGrayScale = np.zeros((testsQuantity, 1025))
    for i in range(0, len(imgs)):
        imgsGrayScale[i] = np.append(gray_scale(imgs[i]), [1])
    return [imgsGrayScale, labels[:testsQuantity]]


def iris_loader():
    result = loadIRIS(35, True)
    return result


def iris_loader_testing(testsQuantity):
    return loadIRIS(35, False, testsQuantity)


# Getting acceptable solution
def stop_condition(genValidation, loss, thresholdLoss, thresholdCorrectness, correctnessLog):
    if thresholdLoss >= loss[0] or thresholdCorrectness <= correctnessLog[0]:
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


def correctness_validation(loadTestData, correctnessLog, generationBestSVM):
    test_data, labels = loadTestData(500)
    classification = np.dot(test_data[:], np.transpose(generationBestSVM))
    sortedClassification = np.argsort(classification[:])
    testDataSize = len(sortedClassification[0]) - 1
    labelsClassification = sortedClassification[:, testDataSize]
    result = (np.sum(np.array(labels[:] == labelsClassification[:], dtype='int'))/len(labelsClassification))*100
    return np.append([result], correctnessLog)


# data visualization
def data_visualization(lossResults, correctnessResults, SVM):
    plt.plot(range(0, len(lossResults)), lossResults)
    plt.plot(range(0, len(correctnessResults)), correctnessResults)
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.xlabel("Generation")
    plt.show()
    plt.plot(range(0, len(correctnessResults)), correctnessResults)
    plt.show()
    displayImgs(SVM)


def displayImgs(SVM):
    # Display the image, with its name and label in one window you should close it to watch the next one
    fig = plt.figure(figsize=(32, 32))
    columns = 2
    rows = 2
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(SVM[i-1][:len(SVM[i-1])-1].reshape(32, 32), cmap=cm.gray)
    plt.show()


def loss_index(adaptability, lossLog, population, data, quantity):
    adaptabilitySelectionIndex, generationBetterLoss = adaptability(population, data, quantity)
    result = np.array([population[adaptabilitySelectionIndex[0]]])
    for i in range(1, len(adaptabilitySelectionIndex)):
        result = np.append(result, np.array([population[adaptabilitySelectionIndex[i]]]), axis=0)
    return result, np.append([generationBetterLoss], lossLog)


# linear classification model
def linear_model_optimization(probabilityDistribution, initializer, matching, mutate, crossing, crossover, adaptability,
                              loadData, genValidation, correctnessAnalysis, correctnessValidation, loadTestData,
                              optimalIndividual, thresholdLoss, thresholdCorrectness, nValidationGeneration, populationSize, scaleGeneration, bins):
    lossLog = []
    correctnessLog = []
    data = loadData()
    population = initializer(probabilityDistribution, len(data[0][0]), populationSize, scaleGeneration, bins)
    OptimalSVMLog, lossLog = loss_index(adaptability, lossLog, population, data, optimalIndividual)
    correctnessLog = correctnessValidation(loadTestData, correctnessLog, OptimalSVMLog[0])
    generation = 1
    while genValidation(nValidationGeneration, lossLog, thresholdLoss, thresholdCorrectness, correctnessLog):
        print("Generation:", generation, "Best SVM loss:", lossLog[0], "Best SVM Correctness:", correctnessLog[0],
              "Tama?o:", len(population))
        population = crossing(mutate, crossover, matching(OptimalSVMLog))
        population = np.append(population, OptimalSVMLog, axis=0)
        OptimalSVMLog, lossLog = loss_index(adaptability, lossLog, population, data, optimalIndividual)
        correctnessLog = correctnessValidation(loadTestData, correctnessLog, OptimalSVMLog[0])
        generation += 1
    correctnessAnalysis(lossLog, correctnessLog, OptimalSVMLog[0])

## Chi2 + CIFAR
#linear_model_optimization(chi2.rvs, set_up_population, fittest_selection_matching, mutate_population, crossing,
#                          crossover, linear_classifier, load_cifar10, stop_condition, data_visualization,
#                          correctness_validation, load_correctness_data, 16, 84, 90, 4, 160, 256, 4)

## Chi2 + IRIS
#linear_model_optimization(chi2.rvs, set_up_population, fittest_selection_matching, mutate_population, crossing,
#                          crossover, linear_classifier, iris_loader, stop_condition, data_visualization,
#                          correctness_validation, iris_loader_testing, 16, 0.1, 98, 4, 160, 256, 4)


#---- Normal-----#
#linear_model_optimization(norm.rvs, set_up_populationNorm, matching, mutationAlgthm,
#                          crossingNorm, crossoverNorm, linear_classifier, load_cifar10, stop_condition,
#                          data_visualization, correctness_validation, load_correctness_data, 16, 84, 90, 4, 160, 256, 4)
