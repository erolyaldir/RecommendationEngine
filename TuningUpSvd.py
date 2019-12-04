import random
import numpy as np
from DataImporter import DataImporter
from surprise import SVD,NormalPredictor 
from BussinessLayer import BussinessLayer
from surprise.model_selection import GridSearchCV



def LoadDataset():
    ml = DataImporter()
    print("Loading movie ratings...")
    data = ml.loadDatasetFromFile()
    print("\nComputing movie popularity ranks")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadDataset()

print("Searching for best parameters...")
param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010],'n_factors': [50, 100]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(evaluationData)

# best RMSE score
print("Best RMSE score : ", gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

#Create Algortihm
evaluator = BussinessLayer(evaluationData, rankings)

params = gs.best_params['rmse']
SVDtuned = SVD(n_epochs = params['n_epochs'], lr_all = params['lr_all'], n_factors = params['n_factors'])
evaluator.InsertAlgortihm(SVDtuned, "SVD - Tuned")

SVDUntuned = SVD()
evaluator.InsertAlgortihm(SVDUntuned, "SVD - Untuned")

# Add normal recommendation for calculate
Random = NormalPredictor()
evaluator.InsertAlgortihm(Random, "Random")

# calculating Scores
evaluator.ProcessAlgorithm(False)

evaluator.SampleTopNRecs(ml)
