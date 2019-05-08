// <SnippetAddUsings>
using System;
using Microsoft.ML.Data;
// </SnippetAddUsings>

namespace SentimentAnalysis
{
    // <SnippetDeclareTypes>
    public class SentimentData
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }

    public class trade
    {
        

        //public int id { get; set; }
        //public string code { get; set; }
        //public DateTime time_stamp { get; set; }
        //public int profitMul { get; set; }
        public float sma200Dist { get; set; }  // positve means follow trade direction
        public float sma50Dist { get; set; }   // positve means follow trade direction
        public float sma21Dist { get; set; }   // positve means follow trade direction
        //public double stopLossPips { get; set; }

        public string position { get; set; }
        //public int openDur { get; set; }
        public float barRatio { get; set; }
        public float NumOfReverseBars { get; set; }
        public float bolUpDist { get; set; }
        public float bolDownDist { get; set; }
        public float sma200Slope { get; set; }  // positve means follow trade direction
        public float sma50Slope { get; set; }   // positve means follow trade direction
        public float sma21Slope { get; set; }   // positve means follow trade direction
        public float bodyRatio { get; set; }
        public float smaPos { get; set; }

        [ColumnName("Label")]
        public bool result { get; set; }
    }

    public class TradePrediction : trade
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }

    }


    public class SentimentPrediction : SentimentData
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
    // </SnippetDeclareTypes>
}
