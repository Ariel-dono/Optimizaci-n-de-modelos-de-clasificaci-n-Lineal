# Imports
from data_loading import *
from scipy.stats import chi2
from scipy.stats import norm
import random



# randomly generate the first generation
def set_up_population(probabilityDistribution, individualSize, populationSize, scaleGeneration, bins):
    population = np.array([np.array(probabilityDistribution(int(scaleGeneration/2), size=[bins, individualSize]), dtype=int)])
    for i in range(1, populationSize):
        population = np.append(population, [np.array(probabilityDistribution(int(scaleGeneration/2), size=[bins, individualSize]), dtype=int)], axis=0)
    return population


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


linear_model_optimization(chi2.rvs, set_up_population, fittest_selection_matching, mutate_population,
                          crossing, crossover, linear_classifier, loadCIFAR10, stop_condition,
                          data_visualization, 16, 5, 4, 160, 256, 4)
