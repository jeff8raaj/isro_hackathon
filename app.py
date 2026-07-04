import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import time

# Initialize Dash App with a modern Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Satellite Cloud Removal Dashboard"

# -----------------------------------------------------------------------------
# APP LAYOUT
# -----------------------------------------------------------------------------
app.layout = dbc.Container([
    # Top Header
    dbc.Row([
        dbc.Col(html.H1("🛰️ Satellite Image Cloud Removal Platform", className="text-center my-4 text-primary"), width=12)
    ]),
    
    dbc.Row([
        # COLUMN 1: CONTROLS & SYSTEM STATUS (Left Sidebar)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("⚙️ Settings & Configuration", className="text-light")),
                dbc.CardBody([
                    # Image Upload
                    html.Label("Upload Satellite Image:", className="fw-bold text-info"),
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'borderRadius': '5px', 'textAlign': 'center', 'marginBottom': '15px'
                        },
                        multiple=False
                    ),
                    
                    # Settings
                    html.Label("Model Selection:", className="fw-bold mt-2"),
                    dcc.Dropdown(
                        id='model-select',
                        options=[
                            {'label': 'Cloud-Transformer V2 (High Accuracy)', 'value': 'ct-v2'},
                            {'label': 'Sat-GAN Light (Fast)', 'value': 'sat-gan'},
                            {'label': 'Temporal-ResNet', 'value': 't-resnet'}
                        ],
                        value='ct-v2', className="text-dark mb-3"
                    ),
                    
                    html.Label("Resolution:", className="fw-bold"),
                    dcc.Dropdown(
                        id='resolution-select',
                        options=['Native (10m)', 'Resampled (20m)', 'Downsampled (50m)'],
                        value='Native (10m)', className="text-dark mb-3"
                    ),
                    
                    html.Label("Output Format:", className="fw-bold"),
                    dcc.RadioItems(
                        id='format-select',
                        options=['GeoTIFF', 'PNG', 'JPEG'],
                        value='GeoTIFF',
                        inline=True, inputClassName="me-2", className="mb-3"
                    ),
                    
                    dbc.Button("▶️ Process Image", id="process-btn", color="primary", size="lg", className="w-100 mt-2")
                ])
            ], className="mb-4"),
            
            # System Status
            dbc.Card([
                dbc.CardHeader(html.H4("🖥️ System Status", className="text-light")),
                dbc.CardBody([
                    html.P([html.Strong("Active AI Model: "), html.Span("Cloud-Transformer V2.1.0", className="text-success")]),
                    html.P([html.Strong("GPU Load: "), html.Span("64% (NVIDIA A100)", className="text-warning")]),
                    html.P([html.Strong("CPU Load: "), html.Span("12%", className="text-success")]),
                    html.P([html.Strong("Cloud Storage Available: "), html.Span("1.2 TB / 5.0 TB", className="text-info")])
                ])
            ])
        ], md=3),

        # COLUMN 2: IMAGE VIEWPORT & COMPARISON (Center Panel)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    dbc.Row([
                        dbc.Col(html.H4("🖼️ Interactive Workspace", className="text-light"), md=6),
                        dbc.Col(
                            dcc.Tabs(id="view-tabs", value="compare", children=[
                                dcc.Tab(label="Before / After", value="compare", className="bg-dark text-light"),
                                dcc.Tab(label="Cloud Mask Only", value="mask", className="bg-dark text-light"),
                                dcc.Tab(label="Cloud-Free Output", value="clean", className="bg-dark text-light"),
                            ]), md=6
                        )
                    ])
                ),
                dbc.CardBody([
                    # Image Status Display Bar
                    dbc.Alert("System Idle. Awaiting Satellite Upload...", id="processing-status", color="secondary", className="text-center fw-bold"),
                    
                    # Main Image Container (Simulated with Interactive Zoom Plotly Canvas)
                    dcc.Graph(id='main-image-display', style={'height': '50vh'}),
                    
                    html.Div([
                        html.P("💡 Tip: Use your mouse cursor to drag boxes to Zoom & Inspect specific regions.", className="text-muted text-center mt-2 small"),
                    ]),
                    
                    # Download Controls
                    hr := html.Hr(),
                    html.Div([
                        html.H5("📥 Export & Download Results", className="text-light mb-3"),
                        dbc.ButtonGroup([
                            dbc.Button("Download Cloud-Free Image", color="success", outline=True),
                            dbc.Button("Download Cloud Mask", color="warning", outline=True),
                            dbc.Button("Download Complete PDF Report", color="info", outline=True),
                        ], className="w-100")
                    ], className="mt-3")
                ])
            ])
        ], md=5),

        # COLUMN 3: PERFORMANCE ANALYTICS & METRICS (Right Sidebar)
        dbc.Col([
            # Analytics Card
            dbc.Card([
                dbc.CardHeader(html.H4("📊 Analytics Dashboard", className="text-light")),
                dbc.CardBody([
                    html.H6("Cloud Coverage Statistics", className="text-muted"),
                    html.H2(id="cloud-coverage-stat", children="0.0%", className="text-danger fw-bold mb-4"),
                    
                    html.H6("Model Performance", className="text-muted"),
                    html.Table([
                        html.Tr([html.Td("Processing Time:", className="pe-4"), html.Td(id="proc-time", children="--", className="fw-bold text-warning")]),
                        html.Tr([html.Td("Inference Time:"), html.Td(id="inf-time", children="--", className="fw-bold text-warning")]),
                        html.Tr([html.Td("Confidence Score:"), html.Td(id="conf-score", children="--", className="fw-bold text-success")]),
                    ], className="table table-borderless table-sm text-light mb-4"),
                    
                    html.H6("Image Quality Metrics", className="text-muted"),
                    html.Table([
                        html.Tr([html.Td("SSIM:", className="pe-4"), html.Td(id="metric-ssim", children="--", className="fw-bold text-info")]),
                        html.Tr([html.Td("PSNR:"), html.Td(id="metric-psnr", children="--", className="fw-bold text-info")]),
                        html.Tr([html.Td("MAE / RMSE:"), html.Td(id="metric-rmse", children="--", className="fw-bold text-info")]),
                    ], className="table table-borderless table-sm text-light"),
                ])
            ], className="mb-4"),
            
            # Processing History Card
            dbc.Card([
                dbc.CardHeader(html.H5("📜 Processing History Log", className="text-light")),
                dbc.CardBody([
                    html.Ul([
                        html.Li("Job #10243 - Completed (SSIM: 0.94)", className="text-success small"),
                        html.Li("Job #10242 - Completed (SSIM: 0.89)", className="text-success small"),
                        html.Li("Job #10241 - Failed (Corrupted File)", className="text-danger small"),
                    ], className="ps-3")
                ])
            ])
        ], md=4)
    ], className="g-4")
], fluid=True)

# -----------------------------------------------------------------------------
# CALLBACKS (INTERACTIVE LOGIC)
# -----------------------------------------------------------------------------
@app.callback(
    [Output("processing-status", "children"),
     Output("processing-status", "color"),
     Output("cloud-coverage-stat", "children"),
     Output("proc-time", "children"),
     Output("inf-time", "children"),
     Output("conf-score", "children"),
     Output("metric-ssim", "children"),
     Output("metric-psnr", "children"),
     Output("metric-rmse", "children"),
     Output("main-image-display", "figure")],
    [Input("process-btn", "n_clicks")],
    [State("view-tabs", "value"), State("model-select", "value")]
)
def run_pipeline(n_clicks, active_tab, selected_model):
    # Initial state (Prior to processing anything)
    if n_clicks is None:
        # Show a default blank grid placeholder map
        fig = px.scatter(title="Awaiting Image Upload...")
        fig.update_layout(template="plotly_dark")
        return "Idle", "secondary", "0.0%", "--", "--", "--", "--", "--", "--", fig

    # Mock Data to simulate computation pipelines dynamically 
    # dependent on selected models
    time.sleep(0.5) # Fake slight lag
    
    # Generate mock images using analytical matrices for rendering
    if active_tab == "compare":
        title_text = "Split View: Cloudy (Left) vs Deep-Learning Reconstructed Clear Output (Right)"
        z_data = [[1, 2, 3, 10, 10, 10], [2, 3, 4, 10, 10, 10], [3, 4, 5, 10, 10, 10]]
    elif active_tab == "mask":
        title_text = "Binary Segmented Cloud Mask Overlay"
        z_data = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    else:
        title_text = "Synthesized Cloud-Free Imagery (Ready for Export)"
        z_data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]]

    fig = px.imshow(z_data, labels=dict(x="Longitude X", y="Latitude Y"), title=title_text,
                    color_continuous_scale="Viridis" if active_tab != "mask" else "Reds")
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))

    # Output dynamic structural metrics back to panels
    return (
        "✨ Processing Complete!", "success", 
        "42.8 %",          # Cloud Coverage Stats
        "1.42 Seconds",    # Processing Time
        "280 Milliseconds",# Inference Time
        "97.4 %",          # Confidence Score
        "0.942",           # SSIM
        "31.45 dB",        # PSNR
        "0.012 / 0.035",   # MAE / RMSE
        fig                # Plotly Map Figure
    )

if __name__ == '__main__':
    app.run(debug=True)