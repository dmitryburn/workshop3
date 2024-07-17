import numpy as np
import plotly.graph_objs as go
import scipy.stats as stats
import matplotlib.pyplot as plt

class InteractiveVisualizer:
    def __init__(self, predicted, target):
        self.predicted = predicted
        self.target = target

    def plot_static(self, sort_by_target=False, plot_qq=False, second_predicted=None, 
                    predicted_name='Предсказанные значения', second_predicted_name='Вторые предсказанные значения'):
        if sort_by_target:
            sorted_indices = np.argsort(self.target)
            self.predicted = self.predicted[sorted_indices]
            self.target = self.target[sorted_indices]
            if second_predicted is not None:
                second_predicted = second_predicted[sorted_indices]

        trace_target = go.Scatter(
            x=list(range(len(self.target))),
            y=self.target,
            mode='markers',
            name='Фактические значения',
            marker=dict(color='blue', size=10) 
        )

        trace_predicted = go.Scatter(
            x=list(range(len(self.predicted))),
            y=self.predicted,
            mode='markers',
            name=predicted_name,
            marker=dict(symbol='x', size=10, color='red')  
        )

        data = [trace_target, trace_predicted]

        if second_predicted is not None:
            trace_second_predicted = go.Scatter(
                x=list(range(len(second_predicted))),
                y=second_predicted,
                mode='markers',
                name=second_predicted_name,
                marker=dict(symbol='x', size=10, color='green') 
            )
            data.append(trace_second_predicted)

        for i in range(len(self.predicted)):
            data.append(go.Scatter(
                x=[i, i],
                y=[self.target[i], self.predicted[i]],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'), 
                showlegend=False
            ))

        layout = go.Layout(
            title='Сравнение предсказанных и фактических значений',
            xaxis=dict(title='Индекс'),
            yaxis=dict(title='Значение'),
            hovermode='closest'
        )

        fig = go.Figure(data=data, layout=layout)

        fig.show()

        if plot_qq:
            self.plot_qq()

    def plot_qq(self):
        residuals = self.predicted - self.target
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('QQ Plot of Prediction Errors', fontsize=16)
        plt.grid(True)
        plt.show()

    def plot_simple(self, threshold=None, second_predicted=None, 
                    predicted_name='Предсказанные значения', second_predicted_name='Вторые предсказанные значения'):
        if threshold is not None:
            self.predicted = self.predicted[:threshold]
            self.target = self.target[:threshold]
            if second_predicted is not None:
                second_predicted = second_predicted[:threshold]

        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(self.target)), self.target, label='Фактические значения', color='blue', s=100)  
        plt.scatter(range(len(self.predicted)), self.predicted, label=predicted_name, color='red', marker='x', s=300)

        for i in range(len(self.predicted)):
            plt.plot([i, i], [self.target[i], self.predicted[i]], color='red', linestyle='--') 

        if second_predicted is not None:
            plt.scatter(range(len(second_predicted)), second_predicted, label=second_predicted_name, color='green', marker='x', s=300) 

        plt.xlabel('Индекс')
        plt.ylabel('Значение')
        plt.title('Сравнение предсказанных и фактических значений')
        plt.legend()
        plt.grid(True)
        plt.show()