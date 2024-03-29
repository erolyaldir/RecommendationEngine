from DataLayer import DataLayer
from AlgorithmCore import AlgorithmCore

class BussinessLayer:
    
    algorithms = []
    
    def __init__(self, dataset, rankings):
        ed = DataLayer(dataset, rankings)
        self.dataset = ed
        #If you want to work with more than one algortihm ,This method insert methods and work with all algorihm in list and sort their result
    def InsertAlgortihm(self, algorithm, name):
        alg = AlgorithmCore(algorithm, name)
        self.algorithms.append(alg)
        #This method run all algorithm in list.if TopN is true thats means return N count recommended items
        #Basicly RMSE:RootMeanSquaredError shows accurcy that lower value is good result
                #MAE:MeanAbsoluteError another error accurcy calculation that same as RMSE lower value best result
                #Hit Rate is a calculation that similar to leave-one out cross validation.Try to estimate recommendation
                #Novelty is that try to recommend new and popular items
                #Diversity is used for prevent offer always time only most popular items.
    def ProcessAlgorithm(self, doTopN):
        results = {}
        for algorithm in self.algorithms:
            print("Scoring  ", algorithm.GetAlgortihmName())
            results[algorithm.GetAlgortihmName()] = algorithm.ProcessAlgorithm(self.dataset, doTopN)

        print("\n")
        
        if (doTopN):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format("Name Of Algoritm", "RootMeanSq Error", "Mean Absolute Error", "HitRate", "CumulativeHitRate", "Average Reciprocal", "Coverage", "Diversity", "Novelty"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(name, metrics["rootMean"], metrics["meanAbsolute"], metrics["hitRate"], metrics["cumulativeHitRate"], metrics["averageReciprocal"],metrics["coverage"], metrics["diversity"], metrics["novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))
                
     
        #To create sample user rating items.
    def SampleTopNRecs(self, ml, testSubject=85, k=10):
        
        for algo in self.algorithms:
            print("\nName of Recommendation engine", algo.GetAlgortihmName())
            
            print("\nWaiting for recommendation Model")
            trainSet = self.dataset.GetFullTrainSet()
            algo.GetAlgorithmBySelf().fit(trainSet)
            
            print("Waiting for computing")
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
        
            predictions = algo.GetAlgorithmBySelf().test(testSet)
            
            recommendations = []
            
            print("\nResult of :")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                recommendations.append((intMovieID, estimatedRating))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for ratings in recommendations[:10]:
                print(ml.getMovieName(ratings[0]), ratings[1])
                

            
            
    
    