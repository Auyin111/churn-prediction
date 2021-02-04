import plotly.express as px


class ChartVisualizer:

    def __init__(self):
        print("ChartVisualizer object created")

    @staticmethod
    def plot_pie_label_distribution(df_plot, col_counting, col_label_name, title, textposition='inside',
                                  textinfo='percent+label'):

        fig = px.pie(df_plot,
                     values=col_counting,
                     names=col_label_name,
                     title=title,
                     color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_traces(textposition=textposition, textinfo=textinfo)
        fig.show("png")
