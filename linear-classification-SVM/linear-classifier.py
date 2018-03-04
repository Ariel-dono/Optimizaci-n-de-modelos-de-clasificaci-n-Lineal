# Imports


# Linear classification model
def linearModelOptimization(probabilityDistribution, initializer, matching, mutations, dominantSelection, crossover, adaptability, loadData,
                            genOptimal, genValidation, correctnessAnalysis, threshold, nIndividual):
    lossLog = []
    localLossLog = []
    population = initializer(probabilityDistribution)
    data = loadData()
    while genValidation(genOptimal, lossLog, population, threshold, localLossLog):
        localLossLog = adaptability(population, data)
        lossLog = lossLog.append(localLossLog[0])
        population = mutations(dominantSelection(crossover, matching(population, nIndividual)))

    correctnessAnalysis(lossLog)


