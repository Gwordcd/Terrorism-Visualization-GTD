import dash
from dash import dcc, html, Input, Output, callback, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import traceback

# ==================== 安全数据加载 ====================
def safe_read_csv(filepath, **kwargs):
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return None
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        print(f"❌ 读取失败 {filepath}: {e}")
        traceback.print_exc()
        return None

print("📂 加载数据...")
df_region = safe_read_csv('data/hive-1.csv')
df_attack = safe_read_csv('data/hive-2.csv')
df_mena = safe_read_csv('data/hive-3.csv')
country_df = safe_read_csv('data/Mapreduce_1.txt', sep='\t', header=None,
                           names=['rank', 'country', 'events'],
                           skipinitialspace=True, engine='python')
if country_df is not None:
    country_df['country'] = country_df['country'].str.strip()

# 新增数据
df_region_attack = safe_read_csv('data/任务2_全球地区对比.csv')
df_extreme_events = safe_read_csv('data/任务3_高伤亡异常明细.csv', low_memory=False)
df_extreme_compare = safe_read_csv('data/hive-5.csv')

if df_extreme_events is not None:
    required_cols = ['iyear', 'latitude', 'longitude', 'nkill', 'attacktype1_txt', 'country_txt', 'region_txt', 'gname', 'summary']
    available_cols = [c for c in required_cols if c in df_extreme_events.columns]
    df_extreme_events = df_extreme_events[available_cols].copy()
    df_extreme_events.dropna(subset=['latitude', 'longitude', 'nkill'], inplace=True)
    df_extreme_events['nkill'] = pd.to_numeric(df_extreme_events['nkill'], errors='coerce')
    df_extreme_events.dropna(subset=['nkill'], inplace=True)
    df_extreme_events['iyear'] = df_extreme_events['iyear'].astype(int)
    df_extreme_events['marker_size'] = df_extreme_events['nkill'].clip(5, 200)

region_coords = {
    'Australasia & Oceania': {'lat': -25.2744, 'lon': 133.7751},
    'Central America & Caribbean': {'lat': 12.1165, 'lon': -86.5246},
    'East Asia': {'lat': 35.8617, 'lon': 104.1954},
    'Eastern Europe': {'lat': 50.4501, 'lon': 30.5234},
    'Middle East & North Africa': {'lat': 26.8206, 'lon': 30.8025},
    'North America': {'lat': 37.0902, 'lon': -95.7129},
    'South America': {'lat': -14.2350, 'lon': -51.9253},
    'South Asia': {'lat': 20.5937, 'lon': 78.9629},
    'Southeast Asia': {'lat': 14.0583, 'lon': 108.2772},
    'Sub-Saharan Africa': {'lat': -9.1393, 'lon': 18.3956},
    'Western Europe': {'lat': 46.8182, 'lon': 8.2275},
    'Central Asia': {'lat': 45.5019, 'lon': 64.2987}
}

if df_region is not None:
    df_region['lat'] = df_region['region_txt'].map(lambda x: region_coords.get(x, {}).get('lat'))
    df_region['lon'] = df_region['region_txt'].map(lambda x: region_coords.get(x, {}).get('lon'))
    df_region = df_region.dropna(subset=['lat', 'lon'])
    df_region['decade'] = df_region['decade'].astype(int)
    decades = [int(x) for x in sorted(df_region['decade'].unique())]
else:
    decades = []

app = dash.Dash(__name__, title='全球恐怖袭击数据分析仪表盘')
app.config.suppress_callback_exceptions = True
server = app.server

# CSS 部分（与之前相同，省略以节省篇幅，实际使用时应保留）
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * { box-sizing: border-box; }
            body {
                background-color: #f0f2f6;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                margin: 0;
                padding: 0;
                height: 100vh;
                overflow-y: auto;
            }
            .main-container {
                display: flex;
                flex-direction: column;
                min-height: 100vh;
            }
            .dashboard-title {
                text-align: center;
                color: #1e466e;
                margin: 12px 0 10px;
                font-weight: 600;
                font-size: 28px;
            }
            .card {
                background: white;
                border-radius: 16px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                padding: 12px 16px;
                margin-bottom: 16px;
                width: 100%;
            }
            .flex-row {
                display: flex;
                gap: 16px;
                flex-wrap: wrap;
            }
            .map-container { flex: 2; min-width: 55%; }
            .detail-container { flex: 1; min-width: 30%; }
            .full-width { width: 100%; }
            .half-width { flex: 1; min-width: 40%; }
            .slider-container { margin: 0 0 12px 0; padding: 8px 16px; }
            .footer {
                text-align: center;
                color: #6c757d;
                border-top: 1px solid #dee2e6;
                padding: 12px 0 16px;
                margin-top: 16px;
                font-size: 12px;
            }
            .note {
                font-size: 11px;
                color: #6c757d;
                margin-top: 6px;
                text-align: center;
            }
            .tab-content { padding: 8px 20px; max-width: 1600px; margin: 0 auto; flex: 1; }
            .dash-tabs { margin-bottom: 8px; }
            @media (max-width: 1000px) {
                .tab-content { padding: 8px 12px; }
                .flex-row { flex-direction: column; }
            }
        </style>
    </head>
    <body>
        <div class="main-container">
            {%app_entry%}
            <footer>{%config%}{%scripts%}{%renderer%}</footer>
        </div>
    </body>
</html>
'''

app.layout = html.Div([
    html.H1("🌍 全球恐怖袭击数据分析仪表盘", className="dashboard-title"),
    dcc.Tabs(id='tabs', value='tab-decade', children=[
        dcc.Tab(label='📊 总览模式', value='tab-overview'),
        dcc.Tab(label='📅 年代模式', value='tab-decade'),
        dcc.Tab(label='🔪 袭击类型分析', value='tab-attack'),
        dcc.Tab(label='🚨 异常事件与极端分析', value='tab-extreme'),
    ]),
    html.Div(id='tab-content', className="tab-content"),
    html.Div("数据来源：GTD 1970-2021 | 小组：温蕤畅 李家朋 马天云 王宇昊", className="footer"),
    dcc.Store(id='current-map-scale', data=1.0)
])

# ==================== 原有绘图函数 ====================
def create_country_map():
    try:
        if country_df is None or country_df.empty:
            return go.Figure().add_annotation(text="数据加载失败", showarrow=False)
        fig = px.choropleth(country_df, locations='country', locationmode='country names',
                            color='events', hover_name='country', color_continuous_scale='Reds',
                            range_color=[0, country_df['events'].quantile(0.95)],
                            title='各国恐怖袭击事件总数 (1970-2021)')
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title_font_size=16,
                          height=520, geo=dict(showframe=False, showcoastlines=True,
                                               projection_type='equirectangular'))
        return fig
    except Exception as e:
        return go.Figure().add_annotation(text=f"错误: {e}", showarrow=False)

def create_attack_chart():
    try:
        if df_attack is None or df_attack.empty:
            return go.Figure().add_annotation(text="数据不可用", showarrow=False)
        df_sorted = df_attack.sort_values('avg_deaths_per_attack', ascending=False)
        fig = px.bar(df_sorted, x='attack_type', y='avg_deaths_per_attack',
                     color='avg_deaths_per_attack', color_continuous_scale='OrRd',
                     title='不同袭击类型平均单次致死人数',
                     labels={'avg_deaths_per_attack': '平均死亡人数', 'attack_type': '袭击类型'})
        fig.update_layout(xaxis={'categoryorder': 'total descending', 'tickangle': -25},
                          title_font_size=16, height=400, margin=dict(l=10, r=10, t=40, b=40))
        return fig
    except Exception as e:
        return go.Figure().add_annotation(text=f"错误: {e}", showarrow=False)

def create_mena_chart():
    try:
        if df_mena is None or df_mena.empty:
            return go.Figure().add_annotation(text="数据不可用", showarrow=False)
        fig = px.line(df_mena, x='decade', y='event_count', color='top_attack_type',
                      title='中东和北非地区主要袭击类型演变', markers=True)
        fig.update_layout(title_font_size=16, height=400, margin=dict(l=10, r=10, t=40, b=40))
        return fig
    except Exception as e:
        return go.Figure().add_annotation(text=f"错误: {e}", showarrow=False)

def create_extreme_map(year):
    if df_extreme_events is None or df_extreme_events.empty:
        return go.Figure().add_annotation(text="无异常事件数据", showarrow=False)
    df_year = df_extreme_events[df_extreme_events['iyear'] == year].copy()
    if df_year.empty:
        return go.Figure().add_annotation(text=f"{year} 年无高伤亡事件数据", showarrow=False)
    # 构建自定义悬停文本
    df_year['hover_text'] = df_year.apply(
        lambda r: f"<b>{r['country_txt']}</b><br>💀 死亡: {r['nkill']}<br>⚔️ 类型: {r['attacktype1_txt']}<br>🏴 组织: {r['gname']}",
        axis=1
    )
    fig = px.scatter_geo(df_year, lat='latitude', lon='longitude',
                         size='marker_size', color='attacktype1_txt',
                         hover_name='hover_text',  # 直接用自定义文本
                         projection='equirectangular',
                         title=f"{year} 年高伤亡异常事件分布",
                         size_max=30, color_discrete_sequence=px.colors.qualitative.Prism)
    fig.update_traces(hovertemplate='%{hovertext}<extra></extra>')
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True),
                      margin=dict(l=0, r=0, t=40, b=0), height=480)
    return fig

def create_treemap():
    if df_region_attack is None or df_region_attack.empty:
        return go.Figure().add_annotation(text="数据不可用", showarrow=False)
    fig = px.treemap(df_region_attack, path=['region_txt', 'attacktype1_txt'],
                     values='袭击次数', color='袭击次数', color_continuous_scale='Blues',
                     title='各地区主要袭击类型事件数（矩形面积=事件数）')
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=480)
    return fig

def create_extreme_rank():
    if df_extreme_compare is None or df_extreme_compare.empty:
        return go.Figure().add_annotation(text="数据不可用", showarrow=False)
    df_rank = df_extreme_compare.sort_values('avg_casualties_per_extreme_event', ascending=False).head(15)
    fig = px.bar(df_rank, y='region_txt', x='avg_casualties_per_extreme_event',
                 color='attacktype_txt', orientation='h',
                 hover_data={'max_casualties_in_group': True},
                 title='极端事件平均伤亡人数排行榜 (区域+袭击类型)',
                 labels={'avg_casualties_per_extreme_event': '平均伤亡人数', 'region_txt': '区域'})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=480, margin=dict(l=0, r=0, t=40, b=0))
    return fig

# ==================== 标签页回调 ====================
@callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-overview':
        return html.Div([
            html.Div(className="card", children=[
                dcc.Graph(id='map-country', figure=create_country_map(),
                          config={'scrollZoom': False})
            ]),
            html.Div("颜色越深表示袭击事件越多。使用右上角按钮缩放地图。", className="note")
        ])
    elif tab == 'tab-decade':
        if df_region is None:
            return html.Div("❌ 数据不可用", className="card")
        marks = {int(d): str(d) for d in decades}
        return html.Div([
            html.Div(className="card slider-container", children=[
                html.Label("🔘 选择年代：", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                dcc.Slider(id='decade-slider', min=int(min(decades)), max=int(max(decades)),
                           step=10, value=2010, marks=marks,
                           tooltip={"placement": "bottom", "always_visible": True})
            ]),
            html.Div(className="flex-row", children=[
                html.Div(className="map-container card", children=[
                    dcc.Graph(id='bubble-map-region', style={'height': '420px'},
                              config={'scrollZoom': False, 'displayModeBar': True})
                ]),
                html.Div(className="detail-container card", children=[
                    html.H5("📌 区域年代详情", style={'marginTop': 0, 'marginBottom': 8}),
                    html.Div("点击左侧地图气泡", style={'color': '#555', 'fontSize': '12px', 'marginBottom': '10px'}),
                    dcc.Graph(id='region-detail-bar', style={'height': '380px'})
                ])
            ]),
            html.Div(className="card full-width", children=[
                dcc.Graph(id='trend-line', style={'height': '280px'})
            ]),
            html.Div("气泡大小=伤亡人数，颜色=事件数。使用地图右上角按钮缩放（只能放大，不能缩小到初始尺寸以下）。", className="note")
        ])
    elif tab == 'tab-attack':
        return html.Div([
            html.Div(className="card", children=[dcc.Graph(id='attack-bar', figure=create_attack_chart())]),
            html.Div(className="card", children=[dcc.Graph(id='mena-trend', figure=create_mena_chart())]),
            html.Div("平均致死人数 = 总死亡数 / 事件数。", className="note")
        ])
    elif tab == 'tab-extreme':
        if df_extreme_events is None:
            return html.Div("❌ 异常事件数据文件缺失", className="card")
        years = sorted(df_extreme_events['iyear'].unique())
        # 强制转换为 Python int
        years_int = [int(y) for y in years]
        year_min = int(min(years_int))
        year_max = int(max(years_int))
        # 生成 marks，只显示部分年份避免拥挤
        marks = {y: str(y) for y in years_int if y % 5 == 0 or y == year_min or y == year_max}
        return html.Div([
            html.Div(className="card slider-container", children=[
                html.Label("📅 选择年份：", style={'fontWeight': 'bold', 'fontSize': '14px'}),
                dcc.Slider(id='year-slider', min=year_min, max=year_max,
                           step=1, value=year_max, marks=marks,
                           tooltip={"placement": "bottom", "always_visible": True})
            ]),
            html.Div(className="card", children=[
                dcc.Graph(id='extreme-map', config={'scrollZoom': False, 'displayModeBar': True})
            ]),
            html.Div(className="flex-row", children=[
                html.Div(className="half-width card", children=[
                    dcc.Graph(id='treemap-region-attack', figure=create_treemap())
                ]),
                html.Div(className="half-width card", children=[
                    dcc.Graph(id='extreme-rank', figure=create_extreme_rank())
                ])
            ]),
            html.Div("地图点大小代表死亡人数，颜色区分袭击类型。右侧图表基于全量数据。", className="note")
        ])
    return html.Div()

# ==================== 原有年代模式回调（保持不变） ====================
@callback(
    Output('bubble-map-region', 'figure', allow_duplicate=True),
    Output('current-map-scale', 'data'),
    Input('decade-slider', 'value'),
    prevent_initial_call=True
)
def update_decade_view(selected_decade):
    try:
        if df_region is None or df_region.empty:
            return go.Figure(), 1.0
        df_filtered = df_region[df_region['decade'] == selected_decade].copy()
        fig_bubble = px.scatter_geo(df_filtered, lat='lat', lon='lon',
                                     size='casualties_total', color='event_count',
                                     hover_name='region_txt', text='region_txt',
                                     projection='equirectangular',
                                     title=f'{selected_decade} 年代各区域伤亡 & 事件',
                                     size_max=40, color_continuous_scale='Plasma')
        fig_bubble.update_layout(geo=dict(showframe=False, showcoastlines=True,
                                          projection_scale=1.0),
                                 title_font_size=14, margin=dict(l=0, r=0, t=30, b=0),
                                 height=400)
        return fig_bubble, 1.0
    except Exception as e:
        err_fig = go.Figure().add_annotation(text=f"更新错误: {e}", showarrow=False)
        return err_fig, 1.0

@callback(
    Output('trend-line', 'figure'),
    Input('decade-slider', 'value')
)
def update_trend_line(_):
    fig_line = px.line(df_region, x='decade', y='event_count', color='region_txt',
                       markers=True, title='各区域恐怖袭击事件数趋势 (1970-2020)',
                       labels={'event_count': '事件数', 'decade': '年代'})
    fig_line.update_layout(
        legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5),
        title_font_size=14, margin=dict(l=0, r=0, t=30, b=50), height=260
    )
    return fig_line

@callback(
    Output('bubble-map-region', 'figure', allow_duplicate=True),
    Input('bubble-map-region', 'relayoutData'),
    State('bubble-map-region', 'figure'),
    State('current-map-scale', 'data'),
    prevent_initial_call=True
)
def limit_scale(relayout_data, current_figure, current_scale):
    if not relayout_data or current_figure is None:
        raise dash.exceptions.PreventUpdate
    if 'geo.projection.scale' in relayout_data:
        new_scale = relayout_data['geo.projection.scale']
        if new_scale < 1.0:
            current_figure['layout']['geo']['projection']['scale'] = 1.0
            return current_figure
    raise dash.exceptions.PreventUpdate

@callback(
    Output('region-detail-bar', 'figure'),
    Input('bubble-map-region', 'clickData'),
    prevent_initial_call=True
)
def update_region_detail(click_data):
    try:
        if click_data and 'points' in click_data:
            point = click_data['points'][0]
            region_name = point.get('hovertext') or point.get('text')
            if region_name and df_region is not None:
                df_sel = df_region[df_region['region_txt'] == region_name].sort_values('decade')
                fig = px.bar(df_sel, x='decade', y='casualties_total',
                             title=f'{region_name} 各年代伤亡人数',
                             labels={'casualties_total': '伤亡总人数', 'decade': '年代'},
                             color='decade', color_continuous_scale='Blues')
                fig.update_layout(title_font_size=14, height=360, margin=dict(l=20, r=20, t=40, b=20))
                return fig
        fig = go.Figure()
        fig.add_annotation(text="👈 点击左侧地图气泡", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=12, color="#6c757d"))
        fig.update_layout(height=360)
        return fig
    except Exception as e:
        return go.Figure().add_annotation(text=f"详情错误: {e}", showarrow=False)

# ==================== 新增异常事件地图回调 ====================
@callback(
    Output('extreme-map', 'figure'),
    Input('year-slider', 'value')
)
def update_extreme_map(selected_year):
    return create_extreme_map(selected_year)

if __name__ == '__main__':
    app.run(debug=True, port=8050)