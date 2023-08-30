# region imports
from AlgorithmImports import *
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
# endregion

class MeasuredBlackShark(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 8, 29)
        self.SetCash(100000)
        self.model_json = self.Download('https://drive.google.com/uc?export=download&id=1Q2KUVpl9RKLWXHl9kvTEkW-76YKZPTxN')
        
        self.json_object = json.loads(self.model_json)
        self.model = tf.keras.models.model_from_json(self.model_json)
        layer_arrays = []
        for layer in self.json_object["weights"]:
            layer_arrays.append(np.array(layer))
        self.model.set_weights(layer_arrays)

        self.symbol = self.AddEquity("GOOGL", Resolution.Daily).Symbol
        self.SetBenchmark(self.symbol)


        self.sc = StandardScaler()
        self.Trend = ""
    def OnData(self, data: Slice):

        prediction = self.GetPrediction()
        if prediction > data[self.symbol].Price * 1.05:
            self.SetHoldings(self.symbol, 1)
        elif prediction < data[self.symbol].Price * .93:
            self.SetHoldings(self.symbol, -1)
        elif prediction < data[self.symbol].Price * .97:
            self.SetHoldings(self.symbol, -.5)

    def GetPrediction(self):

        df = self.History(self.symbol, 40).loc[self.symbol]
        df_change = df[["open"]]
        model_input = []
        for index, row in df_change.tail(30).iterrows():
            model_input.append(np.array(row))
        model_input = np.array(model_input)
        model_input = self.sc.fit_transform(model_input)
        pred = self.sc.inverse_transform(self.model.predict(model_input))

        return pred[0][0]