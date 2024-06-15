import os
import time

import torch
import torch.nn as nn

import numpy as np

from collections import Counter


def calAccuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #print(pred.type(), pred.size())
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print(target.type(), target.size())
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def calAccuracy2(output, target):
    correct = output.eq(target).sum().item()
    total_samples = target.size(0)
    accuracy = correct * 100.0 / total_samples
    return accuracy

def calPluralityVotingAccuracy(predictionVectorsList, target, modelsList=None):
    """Computes the accuracy of the model ensemble's predictions using plurality voting."""
    if modelsList is not None:
        predictionVectorsStack = torch.stack([predictionVectorsList[i] for i in modelsList])
        final_predictions = torch.tensor(plurality_voting(predictionVectorsStack))
    else:
        final_predictions = torch.tensor(plurality_voting(predictionVectorsList))
    return calAccuracy2(final_predictions, target)

def calMajorityVotingAccuracy(predictionVectorsList, target, modelsList=None):
    """Computes the accuracy of the model ensemble's predictions using majority voting."""
    if modelsList is not None:
        predictionVectorsStack = torch.stack([predictionVectorsList[i] for i in modelsList])
        final_predictions = torch.tensor(majority_voting(predictionVectorsStack))
    else:
        final_predictions = torch.tensor(majority_voting(predictionVectorsList))
    return calAccuracy2(final_predictions, target)

def plurality_voting(predictionVectorsList):
    """Computes the final predicted class for each sample using plurality voting."""
    final_predictions = []
    num_classes = predictionVectorsList[0].shape[1]
    for i in range(predictionVectorsList[0].shape[0]):
        total_votes = np.zeros(num_classes)
        for prediction_vectors in predictionVectorsList:
            predicted_class = np.argmax(prediction_vectors[i])
            total_votes[predicted_class] += 1
        final_prediction = np.argmax(total_votes)
        final_predictions.append(final_prediction)
    return np.array(final_predictions)

def majority_voting(predictionVectorsList):
    """Computes the final predicted class for each sample using majority voting."""
    final_predictions = []
    num_classes = predictionVectorsList[0].shape[1]
    for i in range(predictionVectorsList[0].shape[0]):
        total_votes = np.zeros(num_classes)
        for prediction_vectors in predictionVectorsList:
            predicted_class = np.argmax(prediction_vectors[i])
            total_votes[predicted_class] += 1
        max_votes = np.max(total_votes)
        winning_threshold = np.sum(total_votes) / 2
        if max_votes > winning_threshold:
            final_prediction = np.argmax(total_votes)
        else:
            final_prediction = -1
        final_predictions.append(final_prediction)
    return np.array(final_predictions)

def calAveragePredictionVectorAccuracy(predictionVectorsList, target, modelsList=None, topk=(1,)):
    predictionVectorsStack = torch.stack(predictionVectorsList)
    if len(modelsList) > 0:
        predictionVectorsStack = predictionVectorsStack[modelsList,...]
    averagePrediction = torch.mean(predictionVectorsStack, dim=0)
    return calAccuracy(averagePrediction, target, topk)


def calNegativeSamplesSet(predictionVectorsList, target):
    """filter the disagreed samples, return an array of sets"""
    batchSize = target.size(0)
    predictionList = list()
    negativeSamplesSet = list()
    
    for pVL in predictionVectorsList:
        _, pred = pVL.max(dim=1)
        predictionList.append(pred)
        negativeSamplesSet.append(set())
        
    for i in range(batchSize):
        for j,_ in enumerate(predictionList):
            if predictionList[j][i] != target[i]:
                negativeSamplesSet[j].add(i)
    return negativeSamplesSet


def calDisagreementSamplesNoGroundTruth(predictionVectorsList, target):
    """filter the disagreed samples without ground truth"""
    batchSize = target.size(0)
    predictionList = list()
    
    for pVL in predictionVectorsList:
        _, pred = pVL.max(dim=1)
        predictionList.append(pred)
    
    sampleID = list()
    sampleTarget = list()
    predictions = list()
    predVectors = list()
    
    for i in range(batchSize):
        pred = []
        predVect = []
        allAgreed = True
        previousPrediction = -1
        for j, p in enumerate(predictionList):
            pred.append(p[i].item())
            predVect.append(predictionVectorsList[j][i])
            if previousPrediction == -1:
                previousPrediction = p[i]
                continue
            if p[i] != previousPrediction:
                allAgreed = False
        if not allAgreed:
            sampleID.append(i)
            sampleTarget.append(target[i].item())
            predictions.append(pred)
            predVectors.append(predVect)
    return sampleID, sampleTarget, predictions, predVectors


def calDisagreementSamplesOneTargetNegative(predictionVectorsList, target, oneTargetIdx):
    """filter the disagreed samples"""
    batchSize = target.size(0)
    predictionList = list()
    
    for pVL in predictionVectorsList:
        _, pred = pVL.max(dim=1)
        predictionList.append(pred)
    
    # return sampleID, sampleTarget, predictions, predVectors
    sampleID = list()
    sampleTarget = list()
    predictions = list()
    predVectors = list()
    
    for i in range(batchSize):
        pred = []
        predVect = []
        for j, p in enumerate(predictionList):
            pred.append(p[i].item())
            predVect.append(predictionVectorsList[j][i])
        if predictionList[oneTargetIdx][i] != target[i]:
            sampleID.append(i)
            sampleTarget.append(target[i].item())
            predictions.append(pred)
            predVectors.append(predVect)
    return sampleID, sampleTarget, predictions, predVectors


def filterModelsFixed(sampleID, sampleTarget, predictions, predVectors, selectModels):
    filteredPredictions = predictions[:, selectModels]
    #print(filteredPredictions.shape)
    filteredPredVectors = predVectors[:, selectModels]
    return sampleID, sampleTarget, filteredPredictions, filteredPredVectors


from ..groupMetrics import calAllDiversityMetrics

def calFocalDiversityScoresPyTorch(
    oneFocalModel,
    teamModelList,
    negativeSamplesList,
    diversityMetricsList,
    # save time
    crossValidation = True,
    nRandomSamples = 100,
    crossValidationTimes = 3
):
    sampleID, sampleTarget, predictions, predVectors = negativeSamplesList[oneFocalModel]
    teamSampleID, teamSampleTarget, teamPredictions, teamPredVectors = \
            filterModelsFixed(sampleID, sampleTarget, predictions, predVectors, teamModelList) 

    if crossValidation:
        tmpMetrics = list()
        for _ in range(crossValidationTimes):
            randomIdx = np.random.choice(np.arange(teamPredictions.shape[0]), nRandomSamples)        
            tmpMetrics.append(calAllDiversityMetrics(teamPredictions[randomIdx], 
                                                     teamSampleTarget[randomIdx], 
                                                     diversityMetricsList))
        tmpMetrics = np.mean(np.array(tmpMetrics), axis=0)
    else:
        tmpMetrics.append(calAllDiversityMetrics(teamPredictions[randomIdx], 
                                                 teamSampleTarget[randomIdx], 
                                                 diversityMetricsList))
    return {diversityMetricsList[i]:tmpMetrics[i].item()  for i in range(len(tmpMetrics))}

