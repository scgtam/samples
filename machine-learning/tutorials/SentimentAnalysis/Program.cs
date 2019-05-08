// <SnippetAddUsings>
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using mt4;

// </SnippetAddUsings>

namespace SentimentAnalysis
{
    class Program
    {
        // <SnippetDeclareGlobalVariables>
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        // </SnippetDeclareGlobalVariables>

        static void Main(string[] args)
        {
            // Create ML.NET context/local environment - allows you to add steps in order to keep everything together 
            // as you discover the ML.NET trainers and transforms 
            // <SnippetCreateMLContext>
            MLContext mlContext = new MLContext();
            // </SnippetCreateMLContext>

            // <SnippetCallLoadData>
            TrainTestData splitDataView = LoadData(mlContext);
            // </SnippetCallLoadData>


            // <SnippetCallBuildAndTrainModel>
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            // </SnippetCallBuildAndTrainModel>

            // <SnippetCallEvaluate>
            Evaluate(mlContext, model, splitDataView.TestSet);
            // </SnippetCallEvaluate>

            // <SnippetCallUseModelWithSingleItem>
            UseModelWithSingleItem(mlContext, model);
            // </SnippetCallUseModelWithSingleItem>

            // <SnippetCallUseModelWithBatchItems>
            //UseModelWithBatchItems(mlContext, model);
            // </SnippetCallUseModelWithBatchItems>

            Console.WriteLine();
            Console.WriteLine("=============== End of process ===============");
            Console.ReadLine();
        }

        public static TrainTestData LoadData(MLContext mlContext)
        {
            // Note that this case, loading your training data from a file, 
            // is the easiest way to get started, but ML.NET also allows you 
            // to load data from databases or in-memory collections.
            // <SnippetLoadData>
            List<trade> data = getData();
            IDataView dataView = mlContext.Data.LoadFromEnumerable<trade>(data);
            // </SnippetLoadData>

            // You need both a training dataset to train the model and a test dataset to evaluate the model.
            // Split the loaded dataset into train and test datasets
            // Specify test dataset percentage with the `testFraction`parameter
            // <SnippetSplitData>
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            // </SnippetSplitData>

            // <SnippetReturnSplitData>        
            return splitDataView;
            // </SnippetReturnSplitData>           
        }

        private static  List<trade> getData()
        {
            List<trade> data = new List<trade>();
            using (var context = new mt4.barContext())
            {
                var trades = context.trade_result.ToList();
                var bars = context.bar_stat.ToList();
                foreach (var trade in trades)
                {
                    trade temp = new trade();
                    var lastSixBars = bars.Where(x => x.id <= trade.id && x.id >= trade.id - 5);
                    var entryBar = lastSixBars.Where(x => x.id == trade.id).FirstOrDefault();
                    double bodyHeight = 0;
                    foreach (var bar in lastSixBars)
                    {
                        bodyHeight += Math.Abs(bar.close - bar.open);
                    }
                    var entryBarHeight = Math.Abs(entryBar.close - entryBar.open);
                    var avgBodyHeight = bodyHeight / 5;
                    temp.smaPos = (float)0;
                    if (trade.sma200Dist >= 1)
                        temp.smaPos = (float)1;
                    

                    temp.bodyRatio = (float)(entryBarHeight / avgBodyHeight);
                    temp.sma200Dist = trade.sma200Dist;
                    temp.sma50Dist = trade.sma50Dist;
                    temp.sma21Dist = trade.sma21Dist;
                    temp.barRatio = (float)trade.barRatio;
                    temp.position = trade.position;
                    temp.NumOfReverseBars = trade.NumOfReverseBars;
                    temp.result = Convert.ToBoolean(trade.result);
                    temp.sma200Slope = (float)trade.sma200Slope;
                    temp.sma50Slope = (float)trade.sma50Slope;
                    temp.sma21Slope = (float)trade.sma21Slope;
                    temp.bolUpDist = (float)trade.bolUPDist;
                    temp.bolDownDist = (float)trade.bolDownDist;
                    data.Add(temp);
                }
            }
            return data;

        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            // Create a flexible pipeline (composed by a chain of estimators) for creating/training the model.
            // This is used to format and clean the data.  
            // Convert the text column to numeric vectors (Features column) 
            // <SnippetFeaturizeText>
            //var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
            //</SnippetFeaturizeText>
            // append the machine learning task to the estimator
            // <SnippetAddTrainer> 
            //.Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            // </SnippetAddTrainer>


            //"sma200Dist", "sma50Dist", "sma21Dist",
            var dataPrepEstimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Pos", inputColumnName:nameof(trade.position))
                                                        .Append(mlContext.Transforms.NormalizeMinMax("bodyRatio"))
                                                        .Append(mlContext.Transforms.NormalizeMinMax("smaPos"))
                                                        .Append(mlContext.Transforms.NormalizeMinMax("barRatio"))
                                                        .Append(mlContext.Transforms.NormalizeMinMax("sma200Slope"))
                                                        .Append(mlContext.Transforms.NormalizeMinMax("sma50Slope"))
                                                        .Append(mlContext.Transforms.NormalizeMinMax("sma21Slope"))
                                                        .Append(mlContext.Transforms.NormalizeMinMax("bolUpDist"))
                                                        .Append(mlContext.Transforms.NormalizeMinMax("bolDownDist"))
                                                        .Append(mlContext.Transforms.Concatenate("Features",   "bodyRatio", "Pos","smaPos", "barRatio", "sma200Slope", "sma50Slope", "sma21Slope", "bolUpDist", "bolDownDist"))
                                                        .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                                                        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features")); 

            var model = dataPrepEstimator.Fit(splitTrainSet);

            // Create and train the model based on the dataset that has been loaded, transformed.
            // <SnippetTrainModel>
            Console.WriteLine("=============== Create and Train the Model ===============");
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            // </SnippetTrainModel>
            // Returns the model we trained to use for evaluation.
            // <SnippetReturnModel>
            return model;
            // </SnippetReturnModel>
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            // Evaluate the model and show accuracy stats

            //Take the data in, make transformations, output the data. 
            // <SnippetTransformData>
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            // </SnippetTransformData>

            // BinaryClassificationContext.Evaluate returns a BinaryClassificationEvaluator.CalibratedResult
            // that contains the computed overall metrics.
            // <SnippetEvaluate>
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            // </SnippetEvaluate>

            // The Accuracy metric gets the accuracy of a model, which is the proportion 
            // of correct predictions in the test set.

            // The AreaUnderROCCurve metric is equal to the probability that the algorithm ranks
            // a randomly chosen positive instance higher than a randomly chosen negative one
            // (assuming 'positive' ranks higher than 'negative').

            // The F1Score metric gets the model's F1 score.
            // The F1 score is the harmonic mean of precision and recall:
            //  2 * precision * recall / (precision + recall).

            // <SnippetDisplayMetrics>
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}"); 
            Console.WriteLine("=============== End of model evaluation ===============");
            //</SnippetDisplayMetrics>
            
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            // <SnippetCreatePredictionEngine1>
            PredictionEngine<trade, TradePrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<trade, TradePrediction>(model);
            // </SnippetCreatePredictionEngine1>
            using (var context = new mt4.barContext())
            {
                var trade = context.ML_queue.FirstOrDefault();
                trade sampletrade = new trade();


                trade sampleTrade = new trade
                    {
                        position = trade.position,
                        bodyRatio = trade.bodyRatio,
                        sma200Slope = trade.sma200Slope,
                        sma50Slope = trade.sma50Slope,
                        sma21Slope = trade.sma21Slope,
                        bolUpDist = trade.bolUpDist,
                        bolDownDist = trade.bolDownDist,
                        barRatio = trade.barRatio,
                    };
                 </SnippetCreateTestIssue1>

                // <SnippetPredict>
                var resultprediction = predictionFunction.Predict(trade);
                // </SnippetPredict>
                // <SnippetOutputPrediction>
                Console.WriteLine();
                Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

                Console.WriteLine();
                Console.WriteLine($"Sentiment:  | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultprediction.Probability} ");

                Console.WriteLine("=============== End of Predictions ===============");

                Console.WriteLine();
            }
            }
            // </SnippetOutputPrediction>
        }

        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            // Adds some comments to test the trained model's data points.
            // <SnippetCreateTestIssues>
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                }
            };
            // </SnippetCreateTestIssues>

            // Load batch comments just created 
            // <SnippetPrediction>
            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
            // </SnippetPrediction>

            // <SnippetAddInfoMessage>
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            // </SnippetAddInfoMessage>

            Console.WriteLine();
   
            // <SnippetDisplayResults>
            foreach (SentimentPrediction prediction  in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");

            }
            Console.WriteLine("=============== End of predictions ===============");
            // </SnippetDisplayResults>       
        }

        public trade getTrade()
        {
            using (var context = new mt4.barContext())
            {
                var trade = context.ML_queue.FirstOrDefault();
                trade.position = position,
                    bodyRatio = bodyRatio,
                    sma200Slope = sma200Slope,
                    sma50Slope = sma50Slope,
                    sma21Slope = sma21Slope,
                    bolUpDist = bolUpDist,
                    bolDownDist = bolDownDist,
                    barRatio = barRatio,
                }
            }
            return data;
        }


    }
}
