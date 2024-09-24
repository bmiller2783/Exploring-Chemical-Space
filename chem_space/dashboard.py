import sys, base64, datetime, io
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

sys.path.append('/Users/rebecca/Desktop/Coding_Projects/Exploring-Chemical-Space')
from dash import Dash, html, dcc, callback, Output, Input, State
import chem_space as cs
import chem_space.plotting as cs_plot

app = Dash(external_stylesheets=[dbc.themes.MORPH])
app.layout = [
    html.H1(children='Chemical Space', style={'textAlign':'center'}),
    # Parameter sheet upload
    dcc.Upload(id='upload-data',children=html.Div(['Drag and Drop or ',html.A('Select Files')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    # Loading message container
    dbc.Container(id='loading-container',children=[
        dbc.Row([
            dbc.Col([
                html.Div(id='loading-mssg-pca')
            ]),
            dbc.Col([
                html.Div(id='loading-mssg-umap')
            ]),
            dbc.Col([
                html.Div(id='loading-mssg-tsne')
            ])
        ])
    ]
    ),
    # Main container
    dbc.Container([
        dbc.Row([
            html.Div(id='main-div')
        ]),
    ]),
    # Store parsed and computed data
    dcc.Store(id='parsed-store')
]

@callback(
    Output('parsed-store', 'data', allow_duplicate=True),
    Output('loading-mssg-pca', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    prevent_initial_call=True
)
def parse_upload(contents, filename, time):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        _df = df.select_dtypes(include=['object', 'int64'])
        df_ = df.select_dtypes(include=['float32', 'float64'])

        df_.dropna(axis=0, inplace=True)
        df_sc = pd.DataFrame(StandardScaler().fit_transform(df_))
        df_sc.columns = df_.columns
        df = {'Parsed':df_sc.to_dict()}

        mssg = [
            html.H2('Loading PCA (1/3)'),
            dcc.Loading(dbc.Spinner(color="danger"))
        ]

        return df, mssg

@callback(
    Output('parsed-store', 'data', allow_duplicate=True),
    Output('loading-mssg-umap', 'children'),
    Input('loading-mssg-pca', 'children'),
    State('parsed-store', 'data'),
    prevent_initial_call=True
)
def init_pca(mssg, df):
    if mssg:
        df_ = pd.DataFrame(df['Parsed'])
        spc = cs.ChemicalSpace()
        spc.make_space(df_, method='pca')
        df['pca'] = spc.df.to_dict()
        mssg = [
            html.H2('Loading UMAP (2/3)'),
            dcc.Loading(dbc.Spinner(color="danger"))#dbc.Spinner(color="danger")
        ]
        return df, mssg

@callback(
    Output('parsed-store', 'data', allow_duplicate=True),
    Output('loading-mssg-tsne', 'children'),
    Input('loading-mssg-umap', 'children'),
    State('parsed-store', 'data'),
    prevent_initial_call=True
)
def init_umap(mssg, df):
    if mssg:
        df_ = pd.DataFrame(df['Parsed'])
        spc = cs.ChemicalSpace()
        spc.make_space(df_, method='umap')
        df['umap'] = spc.df.to_dict()
        mssg = [
            html.H2('Loading tSNE (3/3)'),
            dcc.Loading(dbc.Spinner(color="danger"))
        ]
        return df, mssg

@callback(
    Output('parsed-store', 'data', allow_duplicate=True),
    Output('loading-container', 'children'),
    Input('loading-mssg-tsne', 'children'),
    State('parsed-store', 'data'),
    prevent_initial_call=True
)
def init_tsne(mssg, df):
    if mssg:
        df_ = pd.DataFrame(df['Parsed'])
        spc = cs.ChemicalSpace()
        spc.make_space(df_, method='tsne')
        df['tsne'] = spc.df.to_dict()
        return df, None

@callback(
    Output('main-div', 'children'),
    Input('loading-container', 'children'),
    State('parsed-store', 'data'),
)
def build(mssg, df_dict):
    if mssg is None:
        df = pd.DataFrame.from_dict(df_dict['pca'], orient='columns')
        clus = cs.Cluster()
        clus.get_clusters(df, method='kmeans')
        df['Clusters'] = clus.labels
        fig = cs_plot.plotly(df)
        fig.update_layout(title='pca + kmeans')

        div = [
            dbc.Container([
                dbc.Col([
                    html.P('Chemical Space Generation:'),
                    dcc.Dropdown(['pca', 'umap', 'tsne'], 'pca', id='dropdown-method', style={'width':'50%'}),
                    html.Div(id='method-par-div', children=[
                        html.P('n-components'),
                        dcc.Input(id='pca-ncomp', type="number"),
                        html.Button(id='pca-btn', children='Recalc')
                    ]),
                    html.P('Clustering Algorithm:'),
                    dcc.Dropdown(['kmeans', 'dbscan', 'hdbscan'], 'kmeans', id='dropdown-cluster-method', style={'width':'50%'}),
                    html.Div(id='cluster-par-div', children=[
                        html.P('K'),
                        dcc.Input(id='kmeans-k', type="number"),
                        html.Button(id='kmeans-btn', children='Recalc'),
                    ]),
                    html.P('Point Size:'),
                    dcc.Slider(5,15,1, value=8, marks=None, id='size-slider', tooltip={"placement":"bottom", "always_visible":False}),
                    html.P('Point Opacity:'),
                    dcc.Slider(0,1,0.1, value=1.0, marks=None, id='opac-slider', tooltip={"placement":"bottom", "always_visible":False})
                ]),
                dbc.Col([
                    html.Div(id='loading-mssg'),
                    dcc.Graph(figure=fig, id='graph-div')
                ])
            ])
        ]
        return div

@callback(
    [
    Output('graph-div', 'figure', allow_duplicate=True),
    Output('method-par-div', 'children'),
    Output('cluster-par-div', 'children')
    ],
    Input('dropdown-method', 'value'),
    Input('dropdown-cluster-method', 'value'),
    State('parsed-store', 'data'),
    prevent_initial_call=True
)
def update_methods(method, cluster, df_dict):
    #if method is not None:
    m, c = None, None
    if method == 'tsne':
        m = [
        html.P('learing rate'),
        dcc.Input(id='tsne-lr', type="number"),
        html.P('perplexity'),
        dcc.Input(id='tsne-perp', type="number"),
        html.Button(id='tsne-btn', children='Recalc')
        ]
    elif method == 'pca':
        m = [
        html.P('n-components'),
        dcc.Input(id='pca-ncomp', type="number"),
        html.Button(id='pca-btn', children='Recalc')
        ]
    elif method == 'umap':
        m = [
        html.P('K-Neighbors'),
        dcc.Input(id='umap-knn', type="number"),
        html.P('Min Distance'),
        dcc.Input(id='umap-d', type="number"),
        html.Button(id='umap-btn', children='Recalc')
        ]

    if cluster == 'kmeans':
        c = [
        html.P('K'),
        dcc.Input(id='kmeans-k', type="number"),
        html.Button(id='kmeans-btn', children='Recalc')
        ]
    elif cluster == 'dbscan':
        c = [
        html.P('epsilon'),
        dcc.Input(id='dbscan-eps', type="number"),
        html.P('min samples'),
        dcc.Input(id='dbscan-min', type="number"),
        html.Button(id='dbscan-btn', children='Recalc')
        ]
    elif cluster == 'hdbscan':
        c = [
        html.P('epsilon'),
        dcc.Input(id='hdbscan-eps', type="number"),
        html.P('min samples'),
        dcc.Input(id='hdbscan-min', type="number"),
        html.Button(id='hdbscan-btn', children='Recalc')
        ]

    df = pd.DataFrame.from_dict(df_dict[method], orient='columns')
    clus = cs.Cluster()
    clus.get_clusters(df, method=cluster)
    df['Clusters'] = clus.labels
    fig = cs_plot.plotly(df, method=method)
    fig.update_layout(title=method+' + '+cluster)
    return fig, m, c

@callback(
    Output('graph-div', 'figure',allow_duplicate=True),
    Input('tsne-btn', 'n_clicks'),
    State('tsne-lr', 'value'),
    State('tsne-perp', 'value'),
    State('parsed-store', 'data'),
    State('dropdown-cluster-method', 'value'),
    prevent_initial_call=True
)
def update_tsne(btn, par1, par2, df_dict, cluster):
    df = pd.DataFrame.from_dict(df_dict['tsne'], orient='columns')
    spc = cs.ChemicalSpace()
    clus = cs.Cluster()
    if par1:
        clus.learning_rate = par1
    if par2:
        clus.perplexity = par2

    spc.make_space(df, method='tsne')
    clus.get_clusters(spc.df, method=cluster)
    spc.df['Clusters'] = clus.labels
    fig = cs_plot.plotly(spc.df, 'tsne')
    return fig

@callback(
    Output('graph-div', 'figure',allow_duplicate=True),
    Input('pca-btn', 'n_clicks'),
    State('pca-ncomp', 'value'),
    State('parsed-store', 'data'),
    State('dropdown-cluster-method', 'value'),
    prevent_initial_call=True
)
def update_pca(btn, par, df_dict, cluster):
    df = pd.DataFrame.from_dict(df_dict['pca'], orient='columns')
    spc = cs.ChemicalSpace()
    clus = cs.Cluster()
    if par:
        clus.n_components = par

    spc.make_space(df, method='pca')
    clus.get_clusters(spc.df, method=cluster)
    spc.df['Clusters'] = clus.labels
    fig = cs_plot.plotly(spc.df, 'pca')
    return fig

@callback(
    Output('graph-div', 'figure', allow_duplicate=True),
    #Output('loading-mssg', 'children', allow_duplicate=True),
    Input('umap-btn', 'n_clicks'),
    State('umap-knn', 'value'),
    State('umap-d', 'value'),
    State('parsed-store', 'data'),
    State('dropdown-cluster-method', 'value'),
    prevent_initial_call=True
)
def update_umap(btn, k, d, df_dict, cluster):
    df = pd.DataFrame.from_dict(df_dict['umap'], orient='columns')
    spc = cs.ChemicalSpace()
    clus = cs.Cluster()
    if k:
        clus.k_nn = k
    if d:
        clus.min_dist = d

    spc.make_space(df, method='umap')
    clus.get_clusters(spc.df, method=cluster)
    spc.df['Clusters'] = clus.labels
    fig = cs_plot.plotly(spc.df, 'umap')
    return fig

@callback(
    Output('graph-div', 'figure', allow_duplicate=True),
    Input('kmeans-btn', 'n_clicks'),
    State('kmeans-k', 'value'),
    State('parsed-store', 'data'),
    State('dropdown-method', 'value'),
    prevent_initial_call=True
)
def update_kmeans(btn, par, df_dict, method):
    df = pd.DataFrame.from_dict(df_dict[method], orient='columns')
    clus = cs.Cluster()
    if par:
        clus.n_clusters = par

    clus.get_clusters(df, method='kmeans')
    df['Clusters'] = clus.labels
    fig = cs_plot.plotly(df, method)
    return fig

@callback(
    Output('graph-div', 'figure',allow_duplicate=True),
    Input('dbscan-btn', 'n_clicks'),
    State('dbscan-eps', 'value'),
    State('dbscan-min', 'value'),
    State('parsed-store', 'data'),
    State('dropdown-method', 'value'),
    prevent_initial_call=True
)
def update_dbscan(btn, par1, par2, df_dict, method):
    df = pd.DataFrame.from_dict(df_dict[method], orient='columns')
    clus = cs.Cluster()
    if par1:
        clus.eps = par1
    if par2:
        clus.min_samples = par2

    clus.get_clusters(df, method='dbscan')
    df['Clusters'] = clus.labels
    fig = cs_plot.plotly(df, method)
    print(fig)
    return fig

@callback(
    Output('graph-div', 'figure', allow_duplicate=True),
    Input('hdbscan-btn', 'n_clicks'),
    State('hdbscan-eps', 'value'),
    State('hdbscan-min', 'value'),
    State('parsed-store', 'data'),
    State('dropdown-method', 'value'),
    prevent_initial_call=True
)
def update_hdbscan(btn, par1, par2, df_dict, method):
    df = pd.DataFrame.from_dict(df_dict[method], orient='columns')
    clus = cs.Cluster()
    if par1:
        clus.eps= par1
    if par2:
        clus.min_samples = par2

    clus.get_clusters(df, method='hdbscan')
    df['Clusters'] = clus.labels
    fig = cs_plot.plotly(df, method)
    return fig

@callback(
    Output('graph-div', 'figure', allow_duplicate=True),
    Input('size-slider', 'value'),
    Input('opac-slider', 'value'),
    State('parsed-store', 'data'),
    State('dropdown-method', 'value'),
    State('dropdown-cluster-method', 'value'),
    prevent_initial_call=True
)
def adjust_plot(size, opac, df_dict, method, cluster):
    df = pd.DataFrame.from_dict(df_dict[method], orient='columns')
    spc = cs.ChemicalSpace()
    clus = cs.Cluster()

    spc.make_space(df, method=method)
    clus.get_clusters(spc.df, method=cluster)
    spc.df['Clusters'] = clus.labels
    fig = cs_plot.plotly(spc.df, method, size, opac)
    return fig

if __name__ == '__main__':
    app.run(debug=True)
