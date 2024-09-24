


def plotly(df, method='pca', s=8, o=1.0):
    import plotly.graph_objects as go
    import plotly.express as px

    fig = go.Figure()
    #colors = {-1:'black', 0:'red', 1:'blue', 2:'grey', 3:'green', 4:'cyan', 5:'orange'}
    #c = df['Clusters'].apply(lambda x: colors[x])
    for c in df['Clusters'].unique():
        df_ = df.loc[df['Clusters']==c]
        fig.add_trace(go.Scatter(x=df_['x'], y=df_['y'], mode='markers',marker=dict(size=s, opacity=o, line=dict(color='black', width=0.4)), name='Cluster '+str(c)))
    return fig

#def mpl(x, y, c=None):
#    import matplotlib.plt as plt



#def bokeh(x, y, c=None):
#    import bokeh
