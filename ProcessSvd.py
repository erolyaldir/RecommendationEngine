import random
import numpy as np
from DataImporter import DataImporter
from surprise import SVD, SVDpp,NormalPredictor 
from BussinessLayer import BussinessLayer 
def LoadDataset():
    ml = DataImporter()
    print("Loading movie ratings...")
    data = ml.loadDatasetFromFile()
    print("\nComputing movie popularity ranks")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)
np.random.seed(0)
random.seed(0)
#Load dataset
(ml, evaluationData, rankings) = LoadDataset()
#Create Algortihm
evaluator = BussinessLayer(evaluationData, rankings)
# SVD
SVD = SVD()
evaluator.InsertAlgortihm(SVD, "SVD")
# SVD++
SVDPlusPlus = SVDpp()
evaluator.InsertAlgortihm(SVDPlusPlus, "SVD++")
# Add normal recommendation for calculate
Random = NormalPredictor()
evaluator.InsertAlgortihm(Random, "Random")
# calculating Scores
evaluator.ProcessAlgorithm(False)
evaluator.SampleTopNRecs(ml)
