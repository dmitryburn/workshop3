import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import scipy.stats as stats

class InteractiveVisualizer:
    def __init__(self, predicted, target):
        self.predicted = predicted
        self.target = target

    def plot_static(self, sort_by_target=False, plot_qq=False):
        if sort_by_target:
            sorted_indices = np.argsort(self.target)
            self.predicted = self.predicted[sorted_indices]
            self.target = self.target[sorted_indices]

        trace_predicted = go.Scatter(
            x=list(range(len(self.target))),
            y=self.predicted,
            mode='lines+markers',
            name='Предсказанные значения'
        )

        trace_target = go.Scatter(
            x=list(range(len(self.target))),
            y=self.target,
            mode='lines+markers',
            name='Фактические значения'
        )

        layout = go.Layout(
            title='Сравнение предсказанных и фактических значений',
            xaxis=dict(title='Индекс'),
            yaxis=dict(title='Значение'),
            hovermode='closest'
        )

        fig = go.Figure(data=[trace_predicted, trace_target], layout=layout)

        fig.show()

        if plot_qq:
            self.plot_qq()

    def plot_qq(self):
        residuals = self.predicted - self.target
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('QQ Plot of Prediction Errors', fontsize=16)
        plt.grid(True)
        plt.show()