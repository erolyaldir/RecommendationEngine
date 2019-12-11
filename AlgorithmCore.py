
from RecConfigrator import RecConfigrator
from DataLayer import DataLayer

class AlgorithmCore:
    
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name
        #This method is main for all calculation.Mainly offer Top10 recommendation.It use inserted all algorithm.
        #Mainly create sample TopN list and try to find HitRate from it.And than calculate cumulativeHitRate.To predict user habit calculate with leave one out.
        #After finding prediction calculate and maintance algortihm find out coverage,diversty and novelty.
    def ProcessAlgorithm(self, evaluationData, doTopN, n=10, verbose=True):
        metrics = {} 
        if (verbose):
            print("Evaluating accuracy...")
        self.algorithm.fit(evaluationData.GetTrainSet())
        predictions = self.algorithm.test(evaluationData.GetTestSet())
        metrics["rootMean"] = RecConfigrator.RMSE(predictions)
        metrics["meanAbsolute"] = RecConfigrator.MAE(predictions)
        
        if (doTopN): 
            if (verbose):
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(evaluationData.GetLOOCVTrainSet())
            leftOutPredictions = self.algorithm.test(evaluationData.GetLOOCVTestSet())     
            allPredictions = self.algorithm.test(evaluationData.GetLOOCVAntiTestSet()) 
            topNPredicted = RecConfigrator.GetTopN(allPredictions, n)
            if (verbose):
                print("Computing hit-rate and rank metrics...") 
            metrics["hitRate"] = RecConfigrator.HitRate(topNPredicted, leftOutPredictions)    
            metrics["cumulativeHitRate"] = RecConfigrator.CumulativeHitRate(topNPredicted, leftOutPredictions) 
            metrics["averageReciprocal"] = RecConfigrator.AverageReciprocalHitRank(topNPredicted, leftOutPredictions)
         
            if (verbose):
                print("Computing recommendations with full data set...")
            self.algorithm.fit(evaluationData.GetFullTrainSet())
            allPredictions = self.algorithm.test(evaluationData.GetFullAntiTestSet())
            topNPredicted = RecConfigrator.GetTopN(allPredictions, n)
            if (verbose):
                print("Analyzing coverage, diversity, and novelty...") 
            metrics["coverage"] = RecConfigrator.UserCoverage( topNPredicted,evaluationData.GetFullTrainSet().n_users,ratingThreshold=4.0)
             
            metrics["diversity"] = RecConfigrator.Diversity(topNPredicted,evaluationData.GetSimilarities())
            
            metrics["novelty"] = RecConfigrator.Novelty(topNPredicted,evaluationData.GetPopularityRankings())
        
        if (verbose):
            print("Analysis complete.")
    
        return metrics
    
    def GetAlgortihmName(self):
        return self.name
    
    def GetAlgorithmBySelf(self):
        return self.algorithm
    
    