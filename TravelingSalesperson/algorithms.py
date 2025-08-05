import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
import random
import math
import heapq
import time
import sys
import csv

def txtToMatrix(filename):
    matrix = np.loadtxt(filename, dtype=float, skiprows=1)
    return matrix

def createAdjMatrix(size):
    adjMatrix = [[0] * size for _ in range(size)]
    for i in range(len(adjMatrix)):
        for j in range(len(adjMatrix[0])):
            if adjMatrix[i][j] == 0 and i != j:
                adjMatrix[i][j] = random.randint(100,1000)
                adjMatrix[j][i] = adjMatrix[i][j]
    
    return np.array(adjMatrix)

def totalDist(adjMatrix, route):
    dist = 0

    for i in range(len(route)):
        dist += adjMatrix[route[i]][route[(i + 1) % len(route)]]
    return dist

def NN(adjMatrix, start, k):
    rwStart = time.time()
    cpuStart = time.process_time()
    visited = [start]
    unvisited = set(range(len(adjMatrix))) - {start}

    currNodes = 0
    while unvisited:
        currNodes += 1
        # Stores all nodes from nearest to furthest, then only keeps the k nearest nodes
        nearest = sorted(unvisited, key=lambda j : adjMatrix[visited[-1]][j])
        nearest = nearest[:k]
        # Choose random from nearest list
        randEdge = random.choice(nearest)

        visited.append(randEdge)
        unvisited.remove(randEdge)
    
    visited.append(start)
    cost = totalDist(adjMatrix, visited)

    cpuStop = time.process_time()
    rwStop = time.time()

    rwTime = rwStop - rwStart
    cpuTime = cpuStop - cpuStart

    return visited, cost, currNodes, rwTime, cpuTime

def NN2O(adjMatrix, route):
    def twoOptSwap(route, v1, v2):
        newRoute = route[: v1 + 1] + route[v1 + 1 : v2 + 1][::-1] + route[v2 + 1 :]
        return newRoute
    
    rwStart = time.time()
    cpuStart = time.process_time()

    bestDist = totalDist(adjMatrix, route)
    currNodes = 0

    improvement = True

    while improvement:
        improvement = False

        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                if j - i == 1 or (i == 0 and j == len(adjMatrix) - 1):
                    continue

                newRoute = twoOptSwap(route, i, j)
                newDist = totalDist(adjMatrix, newRoute)

                currNodes += 1

                if newDist < bestDist:
                    route = newRoute
                    bestDist = newDist
                    improvement = True
                    break
    
    cpuStop = time.process_time()
    rwStop = time.time()

    rwTime = rwStop - rwStart
    cpuTime = cpuStop - cpuStart

    return route, bestDist, currNodes, rwTime, cpuTime

def RNN(adjMatrix, numNearest, numRestarts):
    rwStart = time.time()
    cpuStart = time.process_time()
    route = None
    dist = float('inf')
    currNodes = 0

    for _ in range(numRestarts):
        start = random.randint(0, len(adjMatrix) - 1)

        optRoute, optDist, numNodes, _, _ = NN(adjMatrix, start, numNearest)
        currNodes += numNodes
        optRoute, optDist, numNodes, _, _ = NN2O(adjMatrix, optRoute)
        currNodes += numNodes

        if optDist < dist:
            dist = optDist
            route = optRoute
    
    cpuStop = time.process_time()
    rwStop = time.time()

    rwTime = rwStop - rwStart
    cpuTime = cpuStop - cpuStart

    return route, dist, currNodes, rwTime, cpuTime

def A_MST(adjMatrix):
    def removeAdj(adjMatrix, nodes):
        newAdjMatrix = adjMatrix.copy()

        for node in nodes:
            newAdjMatrix[node, :] = 0
            newAdjMatrix[:, node] = 0

        return newAdjMatrix

    def MST(adjMatrix): 
        mst = minimum_spanning_tree(adjMatrix)

        mst = mst.toarray()
        mstWeight = sum(mst[mst.nonzero()])

        return int(mstWeight)
    
    rwStart = time.time()
    cpuStart = time.process_time()

    resRoute = []
    resDist = float('inf')
    frontier = [(0, 0, None, [], list(range(len(adjMatrix))))]
    heapq.heapify(frontier)

    currNodes = 0

    while frontier:
        fScore, gScore, currNode, visited, remNodes = heapq.heappop(frontier)
        currNodes += 1
        if len(visited) == len(adjMatrix) + 1:
            resDist = gScore
            resRoute = visited
            break
        
        if len(visited) == len(adjMatrix):
            newGScore = gScore + adjMatrix[currNode][visited[0]]
            newFScore = newGScore
            newNode = visited[0]
            newVisited = visited + [visited[0]]
            newRemNodes = []

            heapq.heappush(frontier, (newFScore, newGScore, newNode, newVisited, newRemNodes))
            continue

        for i, node in enumerate(remNodes):
            newGScore = 0 if currNode is None else gScore + adjMatrix[currNode][node]
            
            newMatrix = removeAdj(adjMatrix, visited + [node])

            heuristic = MST(newMatrix)
            newFScore = newGScore + heuristic
            newVisited = visited + [node]
            newRemNodes = remNodes[:i] + remNodes[i + 1:]

            heapq.heappush(frontier, (newFScore, newGScore, node, newVisited, newRemNodes))

    cpuStop = time.process_time()
    rwStop = time.time()

    rwTime = rwStop - rwStart
    cpuTime = cpuStop - cpuStart

    return resRoute, resDist, currNodes, rwTime, cpuTime

def hillClimbing(adjMatrix):
    def getNeighbors(route):
        neighbors = []

        for i in range(len(route)):
            for j in range(i + 1, len(route) - 1):
                neighbor = route[:]

                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbor[-1] = neighbor[0]

                neighbors.append(neighbor)
        
        return neighbors

    rwStart = time.time()
    cpuStart = time.process_time()

    resRoute = list(range(len(adjMatrix)))
    random.shuffle(resRoute)
    resRoute.append(resRoute[0])
    resDist = totalDist(adjMatrix, resRoute)

    improvement = True

    while improvement:
        improvement = False

        neighbors = getNeighbors(resRoute)
        bestNeighbor = None
        bestDist = float('inf')

        for neighbor in neighbors:
            currDist = totalDist(adjMatrix, neighbor)
            if  currDist < bestDist:
                bestDist = currDist
                bestNeighbor = neighbor

        if bestNeighbor is not None and bestDist < resDist:
            resRoute = bestNeighbor
            resDist = bestDist
            improvement = True
        
    cpuStop = time.process_time()
    rwStop = time.time()

    rwTime = rwStop - rwStart
    cpuTime = cpuStop - cpuStart

    return resRoute, resDist, rwTime, cpuTime

def simuAnnealing(adjMatrix, initialTemp, alpha, iterations):
    def getRandomNeighbor(route):
        neighbor = route[:]

        i, j = random.sample(range(1, len(neighbor) - 1), 2)

        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        return neighbor
    
    rwStart = time.time()
    cpuStart = time.process_time()

    resRoute = []
    resDist = float('inf')

    for _ in range(iterations):
        bestRoute = list(range(len(adjMatrix)))
        random.shuffle(bestRoute)
        bestRoute.append(bestRoute[0])
        bestDist = totalDist(adjMatrix, bestRoute)

        T = initialTemp

        while T > 1e-20:
            neighbor = getRandomNeighbor(bestRoute)
            currDist = totalDist(adjMatrix, neighbor)


            if  currDist < bestDist:
                bestDist = currDist
                bestRoute = neighbor
            else:
                acceptProb = math.exp(-(currDist - bestDist) / T)

                if random.random() < acceptProb:
                    bestRoute = neighbor
                    bestDist = currDist
            
            T *= alpha

        if bestDist < resDist:
            resRoute = bestRoute
            resDist = bestDist
    
    cpuStop = time.process_time()
    rwStop = time.time()

    rwTime = rwStop - rwStart
    cpuTime = cpuStop - cpuStart

    return resRoute, resDist, rwTime, cpuTime

def genetic(adjMatrix, numGenerations, mutationProb, popSize):
    def popFitness(population):
        fitness = []
        pop = [currPop for currPop, _, _ in population]
        maxDist = max(totalDist(adjMatrix, route) for route in pop)

        for currPop in pop:
            dist = totalDist(adjMatrix, currPop)
            fitnessCost = maxDist - totalDist(adjMatrix, currPop)
            fitness.append((currPop, dist, fitnessCost))

        fitnessSum = sum(fit for _, _, fit in fitness)

        fitnessProb = [(currPop, dist, (fitnessCost / fitnessSum)) for currPop, dist, fitnessCost in fitness]
        
        sortedPop = sorted(fitnessProb, key= lambda x: x[2], reverse=True)
        
        return sortedPop

    def genChildren(parent1, parent2):
        parent1 = parent1[: len(parent1) - 1]
        parent2 = parent2[: len(parent2) - 1]
        i = random.randint(0, len(parent1) // 2)
        j = random.randint(len(parent1) // 2 + 1, len(parent1) - 1)

        child1 = [len(parent1) + 1] * len(parent1)
        child2 = [len(parent2) + 1] * len(parent2)

        child1[i:j] = parent1[i:j]
        child2[i:j] = parent2[i:j]

        for i in range(len(child1)):
            if child1[i] < len(parent2) + 1:
                continue
            
            if parent2[i] not in child1:
                child1[i] = parent2[i]
            else:
                currIndex = i
                while parent2[currIndex] in child1:
                    for j in range(len(child1)):
                        if child1[j] == parent2[currIndex]:
                            currIndex = j
                            break
                child1[i] = parent2[currIndex]

        for i in range(len(child2)):
            if child2[i] < len(parent1) + 1:
                continue
            
            if parent1[i] not in range(len(child2)):
                child2[i] = parent1[i]
            else:
                currIndex = i
                while parent1[currIndex] in child2:
                    for j in range(len(child2)):
                        if child2[j] == parent1[currIndex]:
                            currIndex = j
                            break
                child2[i] = parent1[currIndex]
        
        child1.append(child1[0])
        child2.append(child2[0])
        return [child1, child2]
    
    def mutate(child):
        i = random.randint(1, len(child) - 2)
        j = random.randint(1, len(child) - 2)

        child[i], child[j] = child[j], child[i]

        return child

    rwStart = time.time()
    cpuStart = time.process_time()

    population = []
    generations = 0

    newPop = list(range(len(adjMatrix)))
    for _ in range(popSize):
        random.shuffle(newPop)

        while newPop in population:
            random.shuffle(newPop)
        newPop.append(newPop[0])
        population.append((newPop, 0, 0))
        newPop = newPop[:len(newPop) - 1]

    population = popFitness(population)

    while generations < numGenerations:
        newPopulation = []
        if population[len(population) - 2][2] == 0:
            i = 0
        for _ in range(len(population) // 2):
            randomParents = random.choices([currPop for currPop, _, _ in population], weights=[float(fitnessProb) for _, _, fitnessProb in population], k=2)

            children = genChildren(list(randomParents[0]), list(randomParents[1]))
            
            for child in children: 
                randProb = random.random()

                if randProb <= mutationProb:
                    child = mutate(child)

                if totalDist(adjMatrix, child) not in [dist for _, dist, _ in population]:
                    newPopulation.append(child)
        
        population = [(currPop, 0, 0) for currPop in newPopulation] + population
        population = sorted(population, key=lambda x : x[1])
        population = population[:popSize]
        population = popFitness(population)
        
        generations += 1
    
    cpuStop = time.process_time()
    rwStop = time.time()

    rwTime = rwStop - rwStart
    cpuTime = cpuStop - cpuStart

    resPop = min(population, key=lambda x : x[1])
    return resPop[0], resPop[1], rwTime, cpuTime
        

def runExperimentsPart1():
    results = {
        'size': [],
        'nnAvgCost': [],
        'nn2oAvgCost': [],
        'rnnAvgCost': [],
        'nnMinCost': [],
        'nn2oMinCost': [],
        'rnnMinCost': [],
        'nnMaxCost': [],
        'nn2oMaxCost': [],
        'rnnMaxCost': [],
        'nnAvgNodes': [],
        'nn2oAvgNodes': [],
        'rnnAvgNodes': [],
        'nnMinNodes': [],
        'nn2oMinNodes': [],
        'rnnMinNodes': [],
        'nnMaxNodes': [],
        'nn2oMaxNodes': [],
        'rnnMaxNodes': [],
        'nnAvgCpu': [],
        'nn2oAvgCpu': [],
        'rnnAvgCpu': [],
        'nnMinCpu': [],
        'nn2oMinCpu': [],
        'rnnMinCpu': [],
        'nnMaxCpu': [],
        'nn2oMaxCpu': [],
        'rnnMaxCpu': [],
        'nnAvgRw': [],
        'nn2oAvgRw': [],
        'rnnAvgRw': [],
        'nnMinRw': [],
        'nn2oMinRw': [],
        'rnnMinRw': [],
        'nnMaxRw': [],
        'nn2oMaxRw': [],
        'rnnMaxRw': [],
    }

    size = 5
    matrices = []
    while size <= 30:
        for _ in range(30):
            matrices.append(createAdjMatrix(size))
        size += 1

    size = 5
    while size <= 30:
        results['size'].append(size)

        nnCosts, nnNodesExp, nnCPU, nnRW = [],[],[],[]
        nnMinCost, nnMinNodes = float('inf'), float('inf')
        nnMaxCost, nnMaxNodes = 0, 0

        nn2oCosts, nn2oNodesExp, nn2oCPU, nn2oRW = [],[],[],[]
        nn2oMinCost, nn2oMinNodes = float('inf'), float('inf')
        nn2oMaxCost, nn2oMaxNodes = 0, 0

        rnnCosts, rnnNodesExp, rnnCPU, rnnRW = [],[],[],[]
        rnnMinCost, rnnMinNodes = float('inf'), float('inf')
        rnnMaxCost, rnnMaxNodes = 0, 0
        for adjMatrix in matrices:
            if len(adjMatrix) != size:
                continue

            _, nnDist, nnNodes, nnRwTime, nnCpuTime = NN(adjMatrix, 0, 1)
            _, nn2oDist, nn2oNodes, nn2oRwTime, nn2oCpuTime = NN2O(adjMatrix, NN(adjMatrix, 0, 1)[0])
            _, rnnDist, rnnNodes, rnnRwTime, rnnCpuTime = RNN(adjMatrix, 3, 10)

            nnRW.append(nnRwTime)
            nnCPU.append(nnCpuTime)
            nnCosts.append(nnDist)
            nnNodesExp.append(nnNodes)

            nn2oRW.append(nn2oRwTime)
            nn2oCPU.append(nn2oCpuTime)
            nn2oCosts.append(nn2oDist)
            nn2oNodesExp.append(nn2oNodes)

            rnnRW.append(rnnRwTime)
            rnnCPU.append(rnnCpuTime)
            rnnCosts.append(rnnDist)
            rnnNodesExp.append(rnnNodes)

            if nnDist < nnMinCost:
                nnMinCost = nnDist
            if nnDist > nnMaxCost:
                nnMaxCost = nnDist
            if nnNodes < nnMinNodes:
                nnMinNodes = nnNodes
            if nnNodes > nnMaxNodes:
                nnMaxNodes = nnNodes

            if nn2oDist < nn2oMinCost:
                nn2oMinCost = nn2oDist
            if nn2oDist > nn2oMaxCost:
                nn2oMaxCost = nn2oDist
            if nn2oNodes < nn2oMinNodes:
                nn2oMinNodes = nn2oNodes
            if nn2oNodes > nn2oMaxNodes:
                nn2oMaxNodes = nn2oNodes

            if rnnDist < rnnMinCost:
                rnnMinCost = rnnDist
            if rnnDist > rnnMaxCost:
                rnnMaxCost = rnnDist
            if rnnNodes < rnnMinNodes:
                rnnMinNodes = rnnNodes
            if rnnNodes > rnnMaxNodes:
                rnnMaxNodes = rnnNodes
            
        results['nnAvgRw'].append(np.mean(nnRW))
        results['nnAvgCpu'].append(np.mean(nnCPU))
        results['nnAvgCost'].append(np.mean(nnCosts))
        results['nnAvgNodes'].append(np.mean(nnNodesExp))
        results['nnMinCost'].append(nnMinCost)
        results['nnMinNodes'].append(nnMinNodes)
        results['nnMaxCost'].append(nnMaxCost)
        results['nnMaxNodes'].append(nnMaxNodes)

        results['nn2oAvgRw'].append(np.mean(nn2oRW))
        results['nn2oAvgCpu'].append(np.mean(nn2oCPU))
        results['nn2oAvgCost'].append(np.mean(nn2oCosts))
        results['nn2oAvgNodes'].append(np.mean(nn2oNodes))
        results['nn2oMinCost'].append(nn2oMinCost)
        results['nn2oMinNodes'].append(nn2oMinNodes)
        results['nn2oMaxCost'].append(nn2oMaxCost)
        results['nn2oMaxNodes'].append(nn2oMaxNodes)

        results['rnnAvgRw'].append(np.mean(rnnRW))
        results['rnnAvgCpu'].append(np.mean(rnnCPU))
        results['rnnAvgCost'].append(np.mean(rnnCosts))
        results['rnnAvgNodes'].append(np.mean(rnnNodes))
        results['rnnMinCost'].append(rnnMinCost)
        results['rnnMinNodes'].append(rnnMinNodes)
        results['rnnMaxCost'].append(rnnMaxCost)
        results['rnnMaxNodes'].append(rnnMaxNodes)

        size += 5
    
    return results

def runExperimentsPart2():
    results = {
        'size': [],
        'nnTotalAvgCostDifference': [],
        'nn2oTotalAvgCostDifference': [],
        'rnnTotalAvgCostDifference': [],
        'nnTotalAvgNodesDifference': [],
        'nn2oTotalAvgNodesDifference': [],
        'rnnTotalAvgNodesDifference': [],
        'nnTotalMinCostDifference': [],
        'nn2oTotalMinCostDifference': [],
        'rnnTotalMinCostDifference': [],
        'nnTotalMinNodesDifference': [],
        'nn2oTotalMinNodesDifference': [],
        'rnnTotalMinNodesDifference': [],
        'nnTotalMaxCostDifference': [],
        'nn2oTotalMaxCostDifference': [],
        'rnnTotalMaxCostDifference': [],
        'nnTotalMaxNodesDifference': [],
        'nn2oTotalMaxNodesDifference': [],
        'rnnTotalMaxNodesDifference': [],
    }

    size = 5
    matrices = []
    while size <= 10:
        for _ in range(30):
            matrices.append(createAdjMatrix(size))
        size += 1

    size = 5
    while size <= 10:
        counter = 0
        results['size'].append(size)

        nnDifference = []
        nn2oDifference = []
        rnnDifference = []
        for adjMatrix in matrices:
            if len(adjMatrix) != size:
                continue

            _, nnDist, nnNodes, _, _ = NN(adjMatrix, 0, 1)
            _, nn2oDist, nn2oNodes, _, _ = NN2O(adjMatrix, NN(adjMatrix, 0, 1)[0])
            _, rnnDist, rnnNodes, _, _ = RNN(adjMatrix, 2, 5)
            _, aStarDist, aStarNodes, _, _ = A_MST(adjMatrix)
            
            nnDifference.append((nnDist - aStarDist, aStarNodes - nnNodes))
            nn2oDifference.append((nn2oDist - aStarDist, aStarNodes - nn2oNodes))
            rnnDifference.append((rnnDist - aStarDist, aStarNodes - rnnNodes))

            counter += 1
            print(counter)
        
        results['nnTotalAvgCostDifference'].append(np.mean([x[0] for x in nnDifference]))
        results['nn2oTotalAvgCostDifference'].append(np.mean([x[0] for x in nn2oDifference]))
        results['rnnTotalAvgCostDifference'].append(np.mean([x[0] for x in rnnDifference]))
        results['nnTotalAvgNodesDifference'].append(np.mean([x[1] for x in nnDifference]))
        results['nn2oTotalAvgNodesDifference'].append(np.mean([x[1] for x in nn2oDifference]))
        results['rnnTotalAvgNodesDifference'].append(np.mean([x[1] for x in rnnDifference]))

        results['nnTotalMinCostDifference'].append(min(nnDifference, key=lambda x : x[0])[0])
        results['nn2oTotalMinCostDifference'].append(min(nn2oDifference, key=lambda x : x[0])[0])
        results['rnnTotalMinCostDifference'].append(min(rnnDifference, key=lambda x : x[0])[0])
        results['nnTotalMinNodesDifference'].append(min(nnDifference, key=lambda x : x[1])[1])
        results['nn2oTotalMinNodesDifference'].append(min(nn2oDifference, key=lambda x : x[1])[1])
        results['rnnTotalMinNodesDifference'].append(min(rnnDifference, key=lambda x : x[1])[1])

        results['nnTotalMaxCostDifference'].append(max(nnDifference, key=lambda x : x[0])[0])
        results['nn2oTotalMaxCostDifference'].append(max(nn2oDifference, key=lambda x : x[0])[0])
        results['rnnTotalMaxCostDifference'].append(max(rnnDifference, key=lambda x : x[0])[0])
        results['nnTotalMaxNodesDifference'].append(max(nnDifference, key=lambda x : x[1])[1])
        results['nn2oTotalMaxNodesDifference'].append(max(nn2oDifference, key=lambda x : x[1])[1])
        results['rnnTotalMaxNodesDifference'].append(max(rnnDifference, key=lambda x : x[1])[1])

        size += 1

    return results

def runExperimentsPart3():
    results = {
        'hillCostDifference': [],
        'simCostDifference': [],
        'genCostDifference': [],
        'hillCpu': [],
        'simCpu': [],
        'genCpu': []
    }

    matrices = []

    for i in range(270,300):
        with open(f"square_graph/graph_size_50_index_{i}.txt", 'r') as file:
            adjMatrix = []
            for line in file:
                row = list(map(float, line.strip().split()))
                if len(row) == 1:
                    continue
                adjMatrix.append(row)
            
            adjMatrix = np.array(adjMatrix)
            matrices.append(adjMatrix)

    hillDifference, hillCPU = [],[]
    simDifference, simCPU = [],[]
    genDifference, genCPU = [],[]
    for adjMatrix in matrices:
        _, hillDist, _, hillCpuTime = hillClimbing(adjMatrix)
        _, simDist, _, simCpuTime = simuAnnealing(adjMatrix, 100000, .99, 5)
        _, genDist, _, genCpuTime = genetic(adjMatrix, 1000, .1, 20)

        hillDifference.append(abs(hillDist - 4))
        simDifference.append(abs(simDist - 4))
        genDifference.append(abs(genDist - 4))

        hillCPU.append(hillCpuTime)
        simCPU.append(simCpuTime)
        genCPU.append(genCpuTime)
        
    results['hillCostDifference'].append(hillDifference)
    results['simCostDifference'].append(simDifference)
    results['genCostDifference'].append(genDifference)

    results['hillCpu'].append(hillCPU)
    results['simCpu'].append(simCPU)
    results['genCpu'].append(genCPU)
    
    
    return results

def plotPart1(results):
    fig, plt1 = plt.subplots(2, 1, figsize=(12,12))

    plt1[0].plot(results['size'], results['nnAvgCost'], marker='o', label='NN', color='b')
    plt1[0].plot(results['size'], results['nn2oAvgCost'], marker='s', label='NN2O', color='g')
    plt1[0].plot(results['size'], results['rnnAvgCost'], marker='d', label='RNN', color='r')
    plt1[0].plot(results['size'], results['nnMinCost'], marker='o', linestyle='--', label='NNMin', color='b')
    plt1[0].plot(results['size'], results['nn2oMinCost'], marker='s', linestyle='--', label='NN2OMin', color='g')
    plt1[0].plot(results['size'], results['rnnMinCost'], marker='d', linestyle='--', label='RNNMin', color='r')
    plt1[0].plot(results['size'], results['nnMaxCost'], marker='o', linestyle='-.', label='NNMax', color='b')
    plt1[0].plot(results['size'], results['nn2oMaxCost'], marker='s', linestyle='-.', label='NN2OMax', color='g')
    plt1[0].plot(results['size'], results['rnnMaxCost'], marker='d', linestyle='-.', label='RNNMax', color='r')
    plt1[0].set_xlabel('Size of Graphs')
    plt1[0].set_ylabel('Total Cost')
    plt1[0].set_title('Total Cost vs. Size of Graphs')
    plt1[0].legend(loc='upper right')
    plt1[0].grid()

    plt1[1].plot(results['size'], results['nnAvgNodes'], marker='o', label='NN', color='b')
    plt1[1].plot(results['size'], results['nn2oAvgNodes'], marker='s', label='NN2O', color='g')
    plt1[1].plot(results['size'], results['rnnAvgNodes'], marker='d', label='RNN', color='r')
    plt1[1].plot(results['size'], results['nnMinNodes'], marker='o', linestyle='--', label='NNMin', color='b')
    plt1[1].plot(results['size'], results['nn2oMinNodes'], marker='s', linestyle='--', label='NN2OMin', color='g')
    plt1[1].plot(results['size'], results['rnnMinNodes'], marker='d', linestyle='--', label='RNNMin', color='r')
    plt1[1].plot(results['size'], results['nnMaxNodes'], marker='o', linestyle='-.', label='NNMax', color='b')
    plt1[1].plot(results['size'], results['nn2oMaxNodes'], marker='s', linestyle='-.', label='NN2OMax', color='g')
    plt1[1].plot(results['size'], results['rnnMaxNodes'], marker='d', linestyle='-.', label='RNNMax', color='r')
    plt1[1].set_xlabel('Size of Graphs')
    plt1[1].set_ylabel('Number of Nodes Visited')
    plt1[1].set_title('Number of Nodes Visited vs. Size of Graphs')
    plt1[1].legend()
    plt1[1].grid()

    plt.tight_layout()
    plt.subplots_adjust(hspace=.5)

    fig, plt2 = plt.subplots(2, 1, figsize=(12,12))

    plt2[0].plot(results['size'], results['nnAvgRw'], marker='o', label='NN', color='b')
    plt2[0].plot(results['size'], results['nn2oAvgRw'], marker='s', label='NN2O', color='g')
    plt2[0].plot(results['size'], results['rnnAvgRw'], marker='d', label='RNN', color='r')
    plt2[0].set_xlabel('Size of Graphs')
    plt2[0].set_ylabel('Average Total Real-World Runtime')
    plt2[0].set_title('Average Real-World Runtimes vs. Size of Graphs')
    plt2[0].legend()
    plt2[0].grid()

    plt2[1].plot(results['size'], results['nnAvgCpu'], marker='o', label='NN', color='b')
    plt2[1].plot(results['size'], results['nn2oAvgCpu'], marker='s', label='NN2O', color='g')
    plt2[1].plot(results['size'], results['rnnAvgCpu'], marker='d', label='RNN', color='r')
    plt2[1].set_xlabel('Size of Graphs')
    plt2[1].set_ylabel('Average Total CPU Runtime')
    plt2[1].set_title('Average CPU Runtimes vs. Size of Graphs')
    plt2[1].legend()
    plt2[1].grid()

    plt.tight_layout()
    plt.subplots_adjust(hspace=.5)
    plt.show()

def plotPart2(results):
    fig, plt1 = plt.subplots(2, 1, figsize=(12,12))

    size = np.array(results['size'])
    width = .05

    plt1[0].bar(size - width * 4, results['nnTotalAvgCostDifference'], width=width, label='NN Avg', color='b')
    plt1[0].bar(size, results['nn2oTotalAvgCostDifference'], width=width, label='NN2O Avg', color='g')
    plt1[0].bar(size + width * 4, results['rnnTotalAvgCostDifference'], width=width, label='RNN Avg', color='y')
    plt1[0].bar(size - width * 5, results['nnTotalMinCostDifference'], width=width, label='NN Min', color='lightblue')
    plt1[0].bar(size - width, results['nn2oTotalMinCostDifference'], width=width, label='NN2O Min', color='lightgreen')
    plt1[0].bar(size + width * 3, results['rnnTotalMinCostDifference'], width=width, label='RNN Min', color='lightcoral')
    plt1[0].bar(size - width * 3, results['nnTotalMaxCostDifference'], width=width, label='NN Max', color='darkblue')
    plt1[0].bar(size + width, results['nn2oTotalMaxCostDifference'], width=width, label='NN2O Max', color='darkgreen')
    plt1[0].bar(size + width * 5, results['rnnTotalMaxCostDifference'], width=width, label='RNN Max', color='darkgoldenrod')
    plt1[0].set_xlabel('Size of Graphs')
    plt1[0].set_ylabel('Total Cost Difference')
    plt1[0].set_title('Total Cost Difference vs. Size of Graphs')
    plt1[0].legend(loc='upper right')
    plt1[0].grid()

    plt1[1].bar(size - width * 4, results['nnTotalAvgNodesDifference'], width=width, label='NN Avg', color='b')
    plt1[1].bar(size, results['nn2oTotalAvgNodesDifference'], width=width, label='NN2O Avg', color='g')
    plt1[1].bar(size + width * 4, results['rnnTotalAvgNodesDifference'], width=width, label='RNN Avg', color='y')
    plt1[1].bar(size - width * 5, results['nnTotalMinNodesDifference'], width=width, label='NN Min', color='lightblue')
    plt1[1].bar(size - width, results['nn2oTotalMinNodesDifference'], width=width, label='NN2O Min', color='lightgreen')
    plt1[1].bar(size + width * 3, results['rnnTotalMinNodesDifference'], width=width, label='RNN Min', color='lightcoral')
    plt1[1].bar(size - width * 3, results['nnTotalMaxNodesDifference'],width=width, label='NN Max', color='darkblue')
    plt1[1].bar(size + width, results['nn2oTotalMaxNodesDifference'], width=width, label='NN2O Max', color='darkgreen')
    plt1[1].bar(size + width * 5, results['rnnTotalMaxNodesDifference'], width=width, label='RNN Max', color='darkgoldenrod')
    plt1[1].set_xlabel('Size of Graphs')
    plt1[1].set_ylabel('Number of Nodes Difference')
    plt1[1].set_title('Number of Nodes Difference vs. Size of Graphs')
    plt1[1].legend(loc='upper left')
    plt1[1].grid()

    plt.tight_layout()
    plt.subplots_adjust(hspace=.25)
    plt.show()

def plotPart3(results):
    fig, plt1 = plt.subplots(1, 1, figsize=(12,12))

    plt1.scatter(results['hillCpu'], results['hillCostDifference'], c='b', label='Hill Climbing')
    plt1.set_xlabel('CPU Runtime')
    plt1.set_ylabel('Total Cost Difference')
    plt1.set_title('Hill Climbing Cost Difference (vs. A*) vs. CPU Runtime')
    plt1.legend(loc='upper right')
    plt1.grid()

    fig, plt2 = plt.subplots(1, 1, figsize=(12,12))

    plt2.scatter(results['simCpu'], results['simCostDifference'], c='g', label='Simulated Annealing')
    plt2.set_xlabel('CPU Runtime')
    plt2.set_ylabel('Total Cost Difference')
    plt2.set_title('Simulated Annealing Cost Difference (vs. A*) vs. CPU Runtime')
    plt2.legend(loc='upper right')
    plt2.grid()

    fig, plt3 = plt.subplots(1, 1, figsize=(12,12))

    plt3.scatter(results['genCpu'], results['genCostDifference'], c='y', label='Genetic Algorithm')
    plt3.set_xlabel('CPU Runtime')
    plt3.set_ylabel('Total Cost Difference')
    plt3.set_title('Genetic Algorithm Cost Difference (vs. A*) vs. CPU Runtime')
    plt3.legend(loc='upper right')
    plt3.grid()

    plt.tight_layout()
    plt.subplots_adjust(hspace=.5)
    plt.show()


def main():
    file = sys.stdin.read()

    adjMatrix = []
    for line in file.strip().split('\n'):
        row = list(map(float, line.split()))
        if len(row) == 1:
            continue
        adjMatrix.append(row)
    
    adjMatrix = np.array(adjMatrix)

    _, nnDist, nnNodes, nnRw, nnCpu = NN(adjMatrix, 0, 1)
    _, nn2oDist, nn2oNodes, nn2oRw, nn2oCpu = NN2O(adjMatrix, NN(adjMatrix, 0, 1)[0])
    _, rnnDist, rnnNodes, rnnRw, rnnCpu = RNN(adjMatrix, 3, 5)
    _, aStarDist, aStarNodes, aStarRw, aStarCpu = A_MST(adjMatrix)
    _, hillDist, hillRw, hillCpu = hillClimbing(adjMatrix)
    _, simDist, simRw, simCpu = simuAnnealing(adjMatrix, 100000, .99, 5)
    _, genDist, genRw, genCpu = genetic(adjMatrix, 1000, .1, 20)

    data = {
        'Nearest Neighbors' : [nnDist, nnNodes, nnCpu, nnRw],
        '2-Opt': [nn2oDist, nn2oNodes, nn2oCpu, nn2oRw],
        'Random Nearest Neighbors': [rnnDist, rnnNodes, rnnCpu, rnnRw],
        'A_MST': [aStarDist, aStarNodes, aStarCpu, aStarRw],
        'Hill Climbing': [hillDist, 0, hillCpu, hillRw],
        'Simulated Annealing': [simDist, 0, simCpu, simRw],
        'Genetic Algorithm': [genDist, 0, genCpu, genRw],
    }

    for algorithm, data in data.items():
        filename = f"{algorithm}.csv"

        with open(filename, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Cost', 'Nodes', 'CPU', 'Real-World'])
            writer.writerow(data)



main()