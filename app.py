'''
    IMPORTS (line 1-19)
'''
import base64
import io
import numpy as np
import pandas as pd
import copy
import os

import plotly
import plotly.graph_objects as go

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table, ctx
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from dash.exceptions import PreventUpdate

'''
    METHODS (line 21-179)
'''
def getIcon(icon: str, height: int=15, size: str=None, color: str=None):
    '''
    returns dash icon with given name
    
    ______________________________ params/returns ______________________________
 
    :param icon: icon name of a dash icon as string 
                 (https://icon-sets.iconify.design/bi/dash/)                                
    :param height: size of dash icon in pixel as int
    :param size: size category for standardized usage (sm, md, lg, xl)

    :return: dash icon
    '''

    if size in ["sm", "md", "lg", "xl"]:
        height={"sm": 15, "md": 20, "lg": 30, "xl": 50}[size]

    if color != None:
        return DashIconify(icon=icon, height=height, color=color)

    return DashIconify(icon=icon, height=height)

def frame_args(duration: int):
    '''
    returns dictionary with frame arguments depend on the given duration
    
    ______________________________ params/returns ______________________________
 
    :param duration: duration per point in animated plot in ms as int                            

    :return: dictionary with arguments
    '''

    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
            }

def getSrcCode(src, job: str, getString: bool=True):
    '''
    reads src file line by line, selects the given job and returns the code as 
    string or pd.DataFrame
    
    ______________________________ params/returns ______________________________
 
    :param src: raw src code as pd.DataFrame with one line per row                         
    :param job: job to filter for in the code
    :param getString: whether to return the code as string or pd.DataFrame

    :return: code as str or pd.DataFrame
    '''

    code = src[src.index > src[src["lines"] == "GLOBAL DEF UP_" + job].index[0]] # filter by start of job
    code = code[code.index < code[code["lines"].str.contains("END ")].index[0]]  # filter by end of job

    if getString == True:
        code_str = ""
        for index, row in code.iterrows():
            code_str = code_str + str(row["lines"]) + "\n"
        return code_str
    else:
        code.reset_index(drop=True, inplace=True)
        return code

def createTimeline(data: pd.DataFrame, getChips: bool=False, theme: str="dark"):
    '''
    creates a list with dmc.TimelineItems from a given pd.DataFrame with 
    subsequent movements
    Displayable movements:
        - PTP, SPTP, LIN, SLIN
        - C_PTP (viapoints) (entry in via column = True)
        - FOR, ENDFOR (number of iterations in point column)
        - WAIT (number of seconds in point column)
    
    ______________________________ params/returns ______________________________
 
    :param data: pd.DataFrame with movements in the following layout:
                 | move | point | via | optimize | X | Y | Z | A | B | C |
                 |------|-------|-----|----------|---|---|---|---|---|---| 
                 | str  |  str  |bool |   bool   |flt|flt|flt|flt|flt|flt|    

    :param getChips: whether to return a list of chips and chosen chips
    :param theme: base theme from mantine provider. Determines the background color

    :return: list of dmc.TimelineItems, 
             (optional: list of dmc.Chips and list of chosen Chip names)
    '''

    if data.loc[0, "move"] != "FOR" and data.loc[0, "move"] != "WAIT":
        if theme == "dark":
            c = "gray.9"
        else:
            c = "gray.1"
        if data.loc[0, "optimize"] == 0:
            timeline = [dmc.TimelineItem(title=dmc.Text(["."], size="sm", color=c), children=[], bullet=dmc.Text(["."], size="sm", color="lime"), bulletSize=1)]  
        else:
            timeline = [dmc.TimelineItem(title=dmc.Text(["."], size="sm", color=c), children=[], bullet=dmc.Text(["."], size="sm", color="green"), bulletSize=1, lineVariant = "dashed")]
    else:
        timeline = []

    chips, chip_values = [], []
    for index, row in data.iterrows():
        if getChips == True:
            # create chips and choose them, if movement is to be optimized
            chips.append(dmc.Chip("optimieren", value="opt_" + str(index), variant="outline"))   
            if row["optimize"] == 1: 
                chip_values.append("opt_" + str(index))

        # show for-loops
        if row["move"] == "FOR":
            if data.loc[index+1, "optimize"] == 1:
                timeline.append(
                    dmc.TimelineItem(
                        title="FOR " + str(row["point"]), lineVariant = "dashed",
                        children=[dmc.Text(["Wiederhole " + str(row["point"]) +  " mal"], color="dimmed", size="sm")]))
            else:
                timeline.append(
                    dmc.TimelineItem(
                        title="FOR " + str(row["point"]), 
                        children=[dmc.Text(["Wiederhole " + str(row["point"]) +  " mal"], color="dimmed", size="sm")]))
        elif row["move"] == "ENDFOR":
            timeline.append(
                dmc.TimelineItem(
                    title="ENDFOR", lineVariant = "dimmed", 
                    children=[dmc.Text(["Ende der Wiederholung"], color="dimmed", size="sm")]))
        # show wait
        elif row["move"] == "WAIT":
            timeline.append(
                dmc.TimelineItem(
                    title="WAIT " + row["point"], lineVariant="dimmed", 
                    children=[dmc.Text(["Warte " + row["point"].split(" ")[1] +  " Sekunden"], color="dimmed", size="sm")]))

        # add points and highlight movements that are to be optimized
        elif index != len(data)-1 and data.loc[index+1, "optimize"] == 1:
            if row["via"] == True:
                timeline.append(
                    dmc.TimelineItem(
                        title=row["point"] + ": VIA", lineVariant = "dashed", 
                        children=[dmc.Text(["X:" + str(row["X"]) + ", Y:" +  str(row["Y"]) + ", Z:" + str(row["Z"]) + 
                                            ", A:" + str(row["A"]) + ", B:" +  str(row["B"]) + ", C:" + str(row["C"])], 
                                           color="dimmed", size="sm")]))
            else:
                timeline.append(
                    dmc.TimelineItem(
                        title=row["point"] + ": " + row["move"], lineVariant = "dashed", 
                        children=[dmc.Text(["X:" + str(row["X"]) + ", Y:" +  str(row["Y"]) + ", Z:" + str(row["Z"]) +
                                            ", A:" + str(row["A"]) + ", B:" +  str(row["B"]) + ", C:" + str(row["C"])], 
                                           color="dimmed", size="sm")]))
        else:
            if row["via"] == True:
                timeline.append(
                    dmc.TimelineItem(
                        title=row["point"] + ": VIA", 
                        children=[dmc.Text(["X:" + str(row["X"]) + ", Y:" +  str(row["Y"]) + ", Z:" + str(row["Z"]) +
                                            ", A:" + str(row["A"]) + ", B:" +  str(row["B"]) + ", C:" + str(row["C"])], 
                                           color="dimmed", size="sm")]))
            else:
                timeline.append(
                    dmc.TimelineItem(
                        title=row["point"] + ": " + row["move"], 
                        children=[dmc.Text(["X:" + str(row["X"]) + ", Y:" +  str(row["Y"]) + ", Z:" + str(row["Z"]) +
                                            ", A:" + str(row["A"]) + ", B:" +  str(row["B"]) + ", C:" + str(row["C"])], 
                                           color="dimmed", size="sm")]))

    if getChips == True:
        return timeline, chips, chip_values
    else:
        return timeline

def createAnimatedPlot(data: pd.DataFrame, theme: str="dark"):
    '''
    creates a animated 3D point plot showing the movements in the given data
    Displayable movements:
        - PTP, SPTP, LIN, SLIN
        - C_PTP (viapoints) (entry in via column = True)
        - FOR, ENDFOR (number of iterations in point column)
        - WAIT (number of seconds in point column)
    
    ______________________________ params/returns ______________________________
 
    :param data: pd.DataFrame with movements in the following layout:
                 | move | point | via | optimize | X | Y | Z | A | B | C |
                 |------|-------|-----|----------|---|---|---|---|---|---| 
                 | str  |  str  |bool |   bool   |flt|flt|flt|flt|flt|flt|   
    :param theme: base theme from mantine provider. Determines the background color

    :return: go.Figure from the animated plot
    '''

    i, loop_active, akt_via = 0, False, None
    x, y, z, optimize, for_i, size = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for index, row in data.iterrows():
        if row["move"] == "FOR":
            if data.loc[index+2, "point"] != akt_via:
                i += 1
            loop_active = True
        elif row["move"] == "ENDFOR":
            loop_active = False
        elif row["move"] == "WAIT":
            pass
        elif len(x) == 0:
            x, y, z = np.append(x, row["X"]), np.append(y, row["Y"]), np.append(z, row["Z"])
            optimize = np.append(optimize, "circle") if row["optimize"] == 0 else np.append(optimize, "diamond")
            size = np.append(size, 20) if row["optimize"] == 0 else np.append(size, 25)
            for_i = np.append(for_i, plotly.colors.qualitative.Dark24[i]) if loop_active == True else np.append(for_i, plotly.colors.qualitative.Dark24[0])
        else:
            x = np.append(x, np.linspace(x[-1], row["X"], 9, endpoint=True)[1:])
            y = np.append(y, np.linspace(y[-1], row["Y"], 9, endpoint=True)[1:])
            z = np.append(z, np.linspace(z[-1], row["Z"], 9, endpoint=True)[1:])
            optimize = np.append(optimize, np.array(["circle"]*8)) if row["optimize"] == 0 else np.append(optimize, np.array(["diamond"]*8))
            size = np.append(size, np.array([20]*8)) if row["optimize"] == 0 else np.append(size, np.array([25]*8))
            if row["via"] == True:
                akt_via = row["point"]
                for_i = np.append(for_i, np.array([plotly.colors.qualitative.Dark24[i]]*7 + [plotly.colors.qualitative.Light24[2]])) if loop_active == True else np.append(for_i, np.array([plotly.colors.qualitative.Dark24[0]]*7 + [plotly.colors.qualitative.Light24[2]]))
            else:
                for_i = np.append(for_i, np.array([plotly.colors.qualitative.Dark24[i]]*8)) if loop_active == True else np.append(for_i, np.array([plotly.colors.qualitative.Dark24[0]]*8))

    # Create figure
    fig = go.Figure(go.Scatter3d(x=[], y=[], z=[], 
                                 mode="markers", marker=dict(color=[], size=[]), opacity=0.9)) # colorscale='Bluered_r'
    
    # Frames
    frames = [go.Frame(data = [go.Scatter3d(x=x[:k+1], y=y[:k+1], z=z[:k+1], marker=dict(color=for_i[:k+1], symbol=optimize[:k+i], size=size[:k+i]))],
                       traces = [0], name = f'frame{k}') for k  in  range(len(x)-1)]

    fig.update(frames=frames)

    sliders = [{"pad": {"b": 10, "t": 60}, "len": 0.9, "x": 0.1, "y": 0, 
                "steps": [{"args": [[f.name], frame_args(0)],
                           "label": str(k),
                           "method": "animate",} for k, f in enumerate(fig.frames)]}]

    fig.update_layout(updatemenus = [{"buttons":[{"args": [None, frame_args(50)], "label": "Start", "method": "animate"},
                                                 {"args": [[None], frame_args(0)], "label": "Pause", "method": "animate"}],
                                      "direction": "left", "pad": {"r": 10, "t": 70}, "type": "buttons", "x": 0.1, "y": 0, "bgcolor": "yellowgreen", "showactive": False}],
                      sliders=sliders)

    fig.update_layout(scene = dict(xaxis=dict(range=[min(x), max(x)], autorange=False),
                                   yaxis=dict(range=[min(y), max(y)], autorange=False),
                                   zaxis=dict(range=[min(z), max(z)], autorange=False)))

    if theme == "dark":
        fig.update_layout(sliders=sliders, width=800, height=800, template="plotly_dark")
    else:
        fig.update_layout(sliders=sliders, width=800, height=800, template="plotly_white")

    return fig

def calcRandomPoint(startp: pd.DataFrame, endp: pd.DataFrame, nvias: int=1):
    '''
    calculates a uniform random point with koordinates in the form x,y,z,a,b,c
    
    ______________________________ params/returns ______________________________
 
    :param startp: row from movement pd.DataFrame corresponding to start of the 
                   movement
    :param endp: row from movement pd.DataFrame corresponding to end of the 
                 movement     
    :param nvias: number of via points         

    :return: random position koordinates x,y,z,a,b,c
    '''

    # calc borders for each via parameter in the form of (min, max)
    X = (startp["X"].values[0] + (endp["X"].values[0]-startp["X"].values[0])/2 - (endp["X"].values[0]-startp["X"].values[0])/5,
         startp["X"].values[0] + (endp["X"].values[0]-startp["X"].values[0])/2 + (endp["X"].values[0]-startp["X"].values[0])/5)
    Y = (startp["Y"].values[0] + (endp["Y"].values[0]-startp["Y"].values[0])/2 - (endp["Y"].values[0]-startp["Y"].values[0])/5,
         startp["Y"].values[0] + (endp["Y"].values[0]-startp["Y"].values[0])/2 + (endp["Y"].values[0]-startp["Y"].values[0])/5)
    Z = (startp["Z"].values[0] + (endp["Z"].values[0]-startp["Z"].values[0])/2 - (endp["Z"].values[0]-startp["Z"].values[0])/5,
         startp["Z"].values[0] + (endp["Z"].values[0]-startp["Z"].values[0])/2 + (endp["Z"].values[0]-startp["Z"].values[0])/5)
    A = (startp["A"].values[0], endp["A"].values[0])
    B = (startp["B"].values[0], endp["B"].values[0])
    C = (startp["C"].values[0], endp["C"].values[0])

    x = np.round(np.random.uniform(X[0], X[1], [nvias]), 3) if nvias != 0 else np.round(np.random.uniform(X[0], X[1]), 3)
    y = np.round(np.random.uniform(Y[0], Y[1], [nvias]), 3) if nvias != 0 else np.round(np.random.uniform(Y[0], Y[1]), 3)
    z = np.round(np.random.uniform(Z[0], Z[1], [nvias]), 3) if nvias != 0 else np.round(np.random.uniform(Z[0], Z[1]), 3)
    a = np.round(np.random.uniform(A[0], A[1], [nvias]), 3) if nvias != 0 else np.round(np.random.uniform(A[0], A[1]), 3)
    b = np.round(np.random.uniform(B[0], B[1], [nvias]), 3) if nvias != 0 else np.round(np.random.uniform(B[0], B[1]), 3)
    c = np.round(np.random.uniform(C[0], C[1], [nvias]), 3) if nvias != 0 else np.round(np.random.uniform(C[0], C[1]), 3)

    return x, y, z, a, b, c

def loadData(data: dict, to_numeric: list=[]):
    '''
    load data from storage, converts it to pd.DataFrame and parses columns to 
    numeric
    
    ______________________________ params/returns ______________________________
 
    :param data: dictionary from storage
    :param to_numeric: names of columns to parse to numeric           

    :return: pd.DataFrame
    '''

    data = pd.DataFrame(data)
    for column in to_numeric:
        data[column] = pd.to_numeric(data[column])
    data.reset_index(drop=True, inplace=True)    

    return data

def createTrialData(data: pd.DataFrame, viaData: dict, trialMethod: str="einfaches Viapunkt-Set", includeNormal: bool=False, nloops: int=1, nvias: int=1):
    '''
    create the trial sequence from the modified job with via points
    
    ______________________________ params/returns ______________________________
 
    :param data: modified job with via points
    :param viaData: dict with informations about via point koordinates
    :param trialMethod: trial method (einfaches Viapunkt-Set, verschienene 
                        Viapunkte or DoE Plan pro Bewegung)   
    :param includeNormal: whether to include the normal state without via points
    :param nloops: number of repitions per via point
    :param nvias: number of different koordinate combinations per via point 

    :return: trial data
    '''

    nvias = 1 if trialMethod == "einfaches Viapunkt-Set" else nvias
    trialData = pd.DataFrame({"move": [], "via": [], "point": [], "X": [], "Y": [], "Z": [], "A": [], "B": [], "C": [], "S": [], "T": [], "optimize": []})

    if trialMethod == "einfaches Viapunkt-Set" or trialMethod == "verschienene Viapunkte":
        if includeNormal:
            trialData = pd.concat([pd.DataFrame({"move": ["FOR"], "via": [False], "point": [nloops], "X": [None], "Y": [None], "Z": [None], "A": [None], "B": [None], "C": [None], "S": [None], "T": [None], "optimize": [0 if data.loc[0,"via"] == False else 1]}), 
                                   data[data["via"] == False], 
                                   pd.DataFrame({"move": ["ENDFOR"], "via": [False], "point": [None], "X": [None], "Y": [None], "Z": [None], "A": [None], "B": [None], "C": [None], "S": [None], "T": [None], "optimize": [0]}),
                                   pd.DataFrame({"move": ["WAIT"], "via": [False], "point": ["SEC 4"], "X": [None], "Y": [None], "Z": [None], "A": [None], "B": [None], "C": [None], "S": [None], "T": [None], "optimize": [0]})])
        for i in range(nvias):
            data_n = copy.deepcopy(data)
            for index, row in data[(data["via"] == True) & (data["optimize"] == 1)].iterrows():
                data_n.loc[index, "X"] = viaData[row["point"]]["X"][i] if row["point"] in viaData.keys() else None
                data_n.loc[index, "Y"] = viaData[row["point"]]["Y"][i] if row["point"] in viaData.keys() else None
                data_n.loc[index, "Z"] = viaData[row["point"]]["Z"][i] if row["point"] in viaData.keys() else None
                data_n.loc[index, "A"] = viaData[row["point"]]["A"][i] if row["point"] in viaData.keys() else None
                data_n.loc[index, "B"] = viaData[row["point"]]["B"][i] if row["point"] in viaData.keys() else None
                data_n.loc[index, "C"] = viaData[row["point"]]["C"][i] if row["point"] in viaData.keys() else None

            trialData = pd.concat([trialData,
                                   pd.DataFrame({"move": ["FOR"], "via": [False], "point": [nloops], "X": [None], "Y": [None], "Z": [None], "A": [None], "B": [None], "C": [None], "S": [None], "T": [None], "optimize": [0 if data.loc[0,"via"] == False else 1]}), 
                                   data_n,
                                   pd.DataFrame({"move": ["ENDFOR"], "via": [False], "point": [None], "X": [None], "Y": [None], "Z": [None], "A": [None], "B": [None], "C": [None], "S": [None], "T": [None], "optimize": [0]}),
                                   pd.DataFrame({"move": ["WAIT"], "via": [False], "point": ["SEC 4"], "X": [None], "Y": [None], "Z": [None], "A": [None], "B": [None], "C": [None], "S": [None], "T": [None], "optimize": [0]})])

    elif trialMethod == "DoE Plan pro Bewegung":
        for index, row in data[(data["via"] == True) & (data["optimize"] == 1)].iterrows():
            if includeNormal:
                trialData = pd.concat([trialData,
                                       pd.DataFrame({"move": ["FOR"], "via": [False], "point": [nloops], "X": [None], "Y": [None], "Z": [None], "A": [None], "B": [None], "C": [None], "S": [None], "T": [None], "optimize": [0]}),
                                       data[index-1 if index!=0 else len(data)-1:index if index!=0 else len(data)],
                                       data[index+1:index+2],
                                       pd.DataFrame({"move": ["ENDFOR"], "via": [False], "point": [None], "X": [None], "Y": [None], "Z": [None], "A": [None], "B": [None], "C": [None], "S": [None], "T": [None], "optimize": [0]}),
                                       pd.DataFrame({"move": ["WAIT"], "via": [False], "point": ["SEC 2"], "X": [None], "Y": [None], "Z": [None], "A": [None], "B": [None], "C": [None], "S": [None], "T": [None], "optimize": [0]})])
            for i in range(nvias):
                x = None if viaData == {} else viaData[row["point"]]["X"][i]
                y = None if viaData == {} else viaData[row["point"]]["Y"][i]
                z = None if viaData == {} else viaData[row["point"]]["Z"][i]
                a = None if viaData == {} else viaData[row["point"]]["A"][i]
                b = None if viaData == {} else viaData[row["point"]]["B"][i]
                c = None if viaData == {} else viaData[row["point"]]["C"][i]
                trialData = pd.concat([trialData,
                                       pd.DataFrame({"move": ["FOR"], "via": [False], "point": [nloops], "X": [None], "Y": [None], "Z": [None], "A": [None], "B": [None], "C": [None], "S": [None], "T": [None], "optimize": [0]}),
                                       data[index-1 if index!=0 else len(data)-1:index if index!=0 else len(data)],
                                       pd.DataFrame({"move": row["move"], "via": [True], "point": row["point"], "X": [x], "Y": [y], "Z": [z], "A": [a], "B": [b], "C": [c], "S": [data.loc[index-1 if index!=0 else index+1,"S"]], "T": [data.loc[index-1 if index!=0 else index+1,"T"]], "optimize": [0]}),
                                       data[index+1:index+2],
                                       pd.DataFrame({"move": ["ENDFOR"], "via": [False], "point": [None], "X": [None], "Y": [None], "Z": [None], "A": [None], "B": [None], "C": [None], "S": [None], "T": [None], "optimize": [0]}),
                                       pd.DataFrame({"move": ["WAIT"], "via": [False], "point": ["SEC 2"], "X": [None], "Y": [None], "Z": [None], "A": [None], "B": [None], "C": [None], "S": [None], "T": [None], "optimize": [0]})])
    else:
        raise "given trial method not found"
    
    trialData.reset_index(drop=True, inplace=True)

    return trialData


'''
    INIT (line 181-188)
'''
# change working directory to app directory
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
if path != os.getcwd():
    os.chdir(path)

'''
    APP

- made with Plotly Dash Manitine


'''

robo_app = dash.Dash(__name__) # create app

# define sidebar layout
sidebar = dmc.Navbar(width={"base": 300},
    children=[
        dmc.NavLink(id="home-link", label="Start", href="/", active=True, variant="subtle", icon=getIcon(icon="bi:house-door-fill", size="sm")),
        dmc.NavLink(id="edit-link", label="Programm editieren", href="/edit", active=False, variant="subtle", icon=getIcon(icon="bi:code-square", size="sm"))
    ]
)

# define content (depends on chosen link in navbar)
content = dmc.Stack(id="page-content")

# define base app layout 
robo_app.layout = dmc.MantineProvider(
    id="mantineProvider",
    withGlobalStyles=True, 
    withNormalizeCSS=True,
    theme={
        "colorScheme": "dark",
        "primaryColor": "lime",
    },
    children=dmc.NotificationsProvider([
        html.Div([
            dcc.Location(id="url"),
            
            dmc.Drawer(title="Menü", children=[sidebar], id="menu-drawer", padding="md", zIndex=10000),
            dmc.Grid([
                dmc.Col([
                    dmc.Burger(id="menu-burger", opened=False, mt=20, ml=20, mr=10)],
                    span="content"), 
                dmc.Divider(orientation="vertical"), 
                dmc.Col(content, span="auto")
            ], justify="flex-start", align="stretch", gutter="xl", mb=110),
            
            dmc.Footer(height=110, fixed=True, withBorder=False,
                children=[
                    dmc.Switch(
                        offLabel=DashIconify(icon="radix-icons:moon", width=17),
                        onLabel=DashIconify(icon="radix-icons:sun", width=17),
                        id="theme-switch", size="lg", checked=False, pl=10)]),

            dmc.Footer(height=70, fixed=True, style={"backgroundColor": "#000000"},
                children=[
                    dmc.Group([
                        dmc.Group([getIcon(icon="simple-icons:mercedes", size="lg", color="white"), dmc.Text("Green Robot", color="white")], position="left"), 
                        dmc.Text("IT-StudentsLab", color="white")], position="apart", p=20)]),

            dcc.Store(id="jobs", storage_type='memory'),
            dcc.Store(id="jobs_tmp1", storage_type='memory'),
            dcc.Store(id="jobs_tmp2", storage_type='memory'),
            dcc.Store(id="jobs_tmp3", storage_type='memory'),

            dcc.Store(id="modjob", storage_type="memory"),
            dcc.Store(id="modjob_tmp1", storage_type="memory"),
            dcc.Store(id="modjob_tmp2", storage_type="memory"),

            dcc.Store(id="vias", storage_type="memory"),
            #dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
        ])
    ], position="top-right", autoClose=5000)
)


# ------ Base Callbacks ------ #

@robo_app.callback(
    output=dict(t = Output("mantineProvider", "theme")),
    inputs=dict(switch = Input("theme-switch", "checked")),
    prevent_initial_call=True
)
def mode(switch):
    if switch == True:
        return dict(t = {"colorScheme": "light", "primaryColor": "lime"})
    else:
        return dict(t = {"colorScheme": "dark", "primaryColor": "lime"})

@robo_app.callback(
    output=dict(d = Output("menu-drawer", "opened"),
                b = Output("menu-burger", 'opened')),
    inputs=dict(drawerOpen = Input("menu-drawer", "opened"),
                burgerOpen = Input("menu-burger", "opened")),
    prevent_initial_call=True
)
def menu(drawerOpen, burgerOpen):
    if ctx.triggered_id == "menu-drawer":
        return dict(d = drawerOpen, b = drawerOpen)
    elif ctx.triggered_id == "menu-burger":
        return dict(d = burgerOpen, b = burgerOpen)
    else:
        raise PreventUpdate

@robo_app.callback(Output('jobs', 'data'),
              Input("jobs_tmp1", 'data'),
              Input("jobs_tmp2", 'data'),
              Input("jobs_tmp3", 'data'))
def update_rawStorage(tmp1, tmp2, tmp3):
    if tmp1 is not None and ctx.triggered_id == "jobs_tmp1":
        return tmp1
    elif tmp2 is not None and ctx.triggered_id == "jobs_tmp2":
        return tmp2  
    elif tmp3 is not None and ctx.triggered_id == "jobs_tmp3":
        return tmp3  
    
    raise PreventUpdate

@robo_app.callback(Output('modjob', 'data'),
              Input("modjob_tmp1", 'data'),
              Input("modjob_tmp2", 'data'))
def update_rawStorage(tmp1, tmp2):
    if tmp1 is not None and ctx.triggered_id == "modjob_tmp1":
        return tmp1
    elif tmp2 is not None and ctx.triggered_id == "modjob_tmp2":
        return tmp2
    
    raise PreventUpdate


# define home page content
home_layout = html.Div(id="home-page", children=[], style={'width': '600px', 'display': 'inline-block', 'vertical-align': 'top', "border-radius": "25px", "margin-right": "35px"}, className="h-100 p-5 bg-light border")

# define edit page content
edit_layout = dmc.Stack([
    dmc.Stepper(
        id="stepper-custom-icons", active=0, breakpoint="sm", 
        children=[
            # Upload page: upload .dat and .src files
            dmc.StepperStep(
                label="Upload", 
                description="Lade das Programm", 
                icon=getIcon(icon="material-symbols:upload-file-outline", size="sm"), 
                progressIcon=getIcon(icon="material-symbols:upload-file-outline", size="sm"), 
                completedIcon=getIcon(icon="material-symbols:upload-file", size="sm"),
                children=[
                    dmc.Divider(size="xs"),
                    dmc.Space(h=20),
                    dmc.Title(f"Upload", order=2, align="center"),
                    dmc.Space(h=20),
                    dmc.Group([
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    dmc.Group([
                                        getIcon(icon="ph:file-code", size="lg"), 
                                        dmc.Text(".src-Datei hochladen", id="job-info", weight=500)], position="left"),
                                    dmc.Badge(" ", id="src-badge", variant="filled", color="red")], position="apart"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350),
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    dmc.Group([
                                        getIcon(icon="mdi:file-table-outline", size="lg"), 
                                        dmc.Text(".dat-Datei hochladen", id="method-info", weight=500)], position="left"),
                                    dmc.Badge(" ", id="dat-badge", variant="filled", color="red")], position="apart"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350)], 
                        position="center", grow=1),
                    dmc.Space(h=20),
                    dmc.Group([
                        dmc.Card([
                            dmc.CardSection(
                                dcc.Upload(id='upload-data', children=dmc.Text("Datei ablegen oder auswählen", weight=500, align="center", h=70, lh=4), multiple=False), 
                                withBorder=True, inheritPadding=True, py="xs")],
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=720)], 
                        position="center", grow=1),
                ]
            ),
            # Potentials page: choose movements to optimize
            dmc.StepperStep(
                label="Potentiale", 
                description="Wähle Bewegungen aus", 
                icon=getIcon(icon="ic:baseline-search", size="sm"), 
                progressIcon=getIcon(icon="ic:baseline-search", size="sm"), 
                completedIcon=getIcon(icon="ic:baseline-saved-search", size="sm"),
                children=[
                    dmc.Divider(size="xs"),
                    dmc.Space(h=20),
                    dmc.Title(f"Potentiale", order=2, align="center"),
                    dmc.Space(h=20),
                    dmc.Group([
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    getIcon(icon="mdi:code-json", size="lg"),
                                    dmc.Text("Job auswählen", id="job-info",weight=500)], position="left"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350),
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    getIcon(icon="mdi:robot-industrial-outline", size="lg"),
                                    dmc.Text("Bewegungsablauf visualisieren", id="method-info", weight=500)], position="left"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350)], 
                        position="center", grow=1),
                    dmc.Space(h=15),
                    dmc.Group([
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    getIcon(icon="wpf:define-location", size="lg"),
                                    dmc.Text("Bewegungen auswählen", id="repeat-info",weight=500)], position="left"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350),
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    getIcon(icon="ic:outline-saved-search", size="lg"),
                                    dmc.Text("Bewegungen suchen", id="variation-info", weight=500)], position="left"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350)], 
                        position="center", grow=1),
                    dmc.Space(h=20),
                    dmc.Group([
                        dmc.SegmentedControl(id="job_input", data=[], radius=20, size="md", maw=720)], 
                        position="center", grow=1),
                    dmc.Space(h=20),
                    dmc.LoadingOverlay(
                        dmc.Accordion(
                            disableChevronRotation=False, 
                            variant="contained", 
                            radius="lg",
                            children=[
                                dmc.AccordionItem([
                                    dmc.AccordionControl("Programmablauf", icon=getIcon(icon="clarity:timeline-line", size="md")),
                                    dmc.AccordionPanel([
                                        dmc.Group([
                                            dmc.Button("Suche Bewegungen", id = "search-button", variant="outline"),
                                            dmc.HoverCard([
                                                dmc.HoverCardTarget(dmc.ActionIcon(getIcon(icon="material-symbols:start-rounded", size="lg"), size="lg", color="gray", variant="subtle", id="GS-button", n_clicks=0)),
                                                dmc.HoverCardDropdown(dmc.Text("Grundstellung anzeigen?", size="sm"))], shadow="md")], 
                                            position="left"),
                                        dmc.Space(h=20), 
                                        dmc.Group([
                                            dmc.Timeline(id="steps-line", active=1, bulletSize=17, lineWidth=3, children=[]),
                                            dmc.ChipGroup(children=[], id="optimize-chips", value=[], multiple=True, position="left", style={"width": "200px", "margin-top": "-20px"}, spacing=38.7)], 
                                            position="left"),
                                    ])
                                ], value="timeline"),
                                dmc.AccordionItem([
                                    dmc.AccordionControl("Visualisierung", icon=getIcon(icon="mdi:robot-industrial-outline", size="md")),
                                    dmc.AccordionPanel(dcc.Graph(id='path-plot')),
                                ], value="plot"),
                                dmc.AccordionItem([
                                    dmc.AccordionControl("Programmcode", icon=getIcon(icon="ic:round-code", size="md")),
                                    dmc.AccordionPanel(dmc.Code(id="raw-code", children="", block=True, style={"width": "650px"}))
                                ], value="code"),
                            ]
                        ), 
                    radius=20),
                ]
            ),
            # Viapoints page: inserts via points, assign koordinates to viapoints
            dmc.StepperStep(
                label="Viapunkte", 
                description="Füge Viapunkte ein",
                icon=getIcon(icon="mdi:graph-timeline-variant", size="sm"), 
                progressIcon=getIcon(icon="mdi:graph-timeline-variant", size="sm"), 
                completedIcon=getIcon(icon="mdi:graph-line-shimmer", size="sm"),
                children=[
                    dmc.Divider(size="xs"),
                    dmc.Space(h=20),
                    dmc.Title(f"Viapunkte", order=2, align="center"),
                    dmc.Space(h=20),
                    dmc.Group([
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    getIcon(icon="iconoir:3d-three-pts-box", size="lg"),
                                    dmc.Text("Versuchsaufbau wählen", id="job-info",weight=500)], position="left"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350),
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    getIcon(icon="fluent:point-scan-24-regular", size="lg"),
                                    dmc.Text("Via-Koordinaten definieren", id="method-info", weight=500)], position="left"),
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350)], 
                        position="center", grow=1),
                    dmc.Space(h=15),
                    dmc.Group([
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    getIcon(icon="ic:round-loop", size="lg"),
                                    dmc.Text("Wiederholungen festlegen", id="repeat-info",weight=500)], position="left"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350),
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    getIcon(icon="material-symbols:format-list-bulleted-add-rounded",size="lg"),
                                    dmc.Text("Variationszahl bestimmen", id="variation-info", weight=500)], position="left"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350)], 
                        position="center", grow=1),
                    dmc.Space(h=20),
                    dmc.Group([
                        dmc.SegmentedControl(id="trial_method", data=["einfaches Viapunkt-Set", "verschienene Viapunkte", "DoE Plan pro Bewegung"], value=None, radius=20, size="md", maw=720)], 
                        position="center", grow=1),
                    dmc.Space(h=20),
                    dmc.LoadingOverlay(
                        dmc.Accordion(
                            disableChevronRotation=False, 
                            variant="contained", 
                            radius="lg",
                            children=[
                                dmc.AccordionItem([
                                    dmc.AccordionControl("Modifizierter Programmablauf", icon=getIcon(icon="clarity:timeline-line", size="md")),
                                    dmc.AccordionPanel([
                                        dmc.Group([
                                            dmc.Button("Viapunkte anpassen", id = "via-button", variant="outline"),
                                            dmc.HoverCard([
                                                dmc.HoverCardTarget(dmc.ActionIcon(getIcon(icon="fad:random-1dice", size="lg"), size="lg", color="gray", variant="outline", id="via-random", n_clicks=0)),
                                                dmc.HoverCardDropdown(dmc.Text("Alle Viapunkte zufällig auswählen?", size="sm"))], shadow="md"),
                                            dmc.NumberInput(id="nloops", value=10, min=1, max=50, step=1, icon=getIcon(icon="ic:round-loop", size="sm"), style={"width": 100}),
                                            dmc.NumberInput(id="nvias", value=2, min=1, max=50, step=1, icon=getIcon(icon="material-symbols:format-list-bulleted-add-rounded", size="sm"), style={"width": 100}),], 
                                            position="left"),
                                        dmc.Space(h=20), 
                                        dmc.Timeline(id="modsteps-line", active=1, bulletSize=17, lineWidth=3, children=[])])], 
                                    value="timeline"),
                                dmc.AccordionItem([
                                    dmc.AccordionControl("Visualisierung", icon=getIcon(icon="mdi:robot-industrial-outline", size="md")),
                                    dmc.AccordionPanel(dcc.Graph(id='modpath-plot'))], 
                                    value="plot"),
                                dmc.AccordionItem([
                                    dmc.AccordionControl("Programmcode", icon=getIcon(icon="ic:round-code", size="md")),
                                    dmc.AccordionPanel(dmc.Code(id="modraw-code", children="", block=True, style={"width": "650px"}))], 
                                    value="code"),
                            ]
                        ),
                    radius=20),
                    dmc.Modal(
                        title="Viapunkte konfigurieren", 
                        id="via-modal", 
                        zIndex=10000, 
                        size=900,
                        children=[
                            dmc.Group([
                                dmc.Select(id="via-select-modal", data=[], value=None, label=None, style={"width": 200}, icon=getIcon(icon="mdi:graph-timeline-variant", size="sm"), rightSection=getIcon(icon="radix-icons:chevron-down", size="sm")),
                                dmc.Group([
                                    dmc.NumberInput(id="via-x-modal", min=None, max=None, step=1, icon=getIcon(icon="mdi:alpha-x", size="md"), style={"width": 150}),
                                    dmc.NumberInput(id="via-y-modal", min=None, max=None, step=1, icon=getIcon(icon="mdi:alpha-y", size="md"), style={"width": 150}),
                                    dmc.NumberInput(id="via-z-modal", min=None, max=None, step=1, icon=getIcon(icon="mdi:alpha-z", size="md"), style={"width": 150}),
                                    dmc.NumberInput(id="via-a-modal", min=None, max=None, step=1, icon=getIcon(icon="mdi:alpha-a", size="md"), style={"width": 150}),
                                    dmc.NumberInput(id="via-b-modal", min=None, max=None, step=1, icon=getIcon(icon="mdi:alpha-b", size="md"), style={"width": 150}),
                                    dmc.NumberInput(id="via-c-modal", min=None, max=None, step=1, icon=getIcon(icon="mdi:alpha-c", size="md"), style={"width": 150})],
                                    position="left", w=500),
                                dmc.HoverCard([
                                    dmc.HoverCardTarget(dmc.ActionIcon(getIcon(icon="fad:random-1dice", size="lg"), size="lg", color="gray", variant="outline", id="via-random-modal", n_clicks=0)),
                                    dmc.HoverCardDropdown(dmc.Text("Zufällig auswählen?", size="sm"))], shadow="md")], 
                                position="left"), 
                            dmc.Space(h=10),
                            dmc.Accordion(
                                disableChevronRotation=False,
                                children=[
                                    dmc.AccordionItem([
                                        dmc.AccordionControl("Visualisierung", icon=getIcon(icon="mdi:robot-industrial-outline", size="md")),
                                        dmc.AccordionPanel(dcc.Graph(id='move-plot', figure=go.Figure(go.Scatter3d(), layout=dict(template="plotly_dark"))))], 
                                        value="plot")],),
                            dmc.Space(h=20), 
                            dmc.Group([
                                dmc.Button("Abbrechen", color="gray", variant="subtle", id="close-button-modal"),
                                dmc.Button("Bestätigen", id="submit-button-modal")], 
                                position="apart")]),
                ]
            ),
            # Summary page: summerizes the results, gives the option to download as txt or as modified .src and .dat
            dmc.StepperCompleted(
                children=[
                    dmc.Divider(size="xs"),
                    dmc.Space(h=20),
                    dmc.Title(f"Zusammenfassung", order=2, align="center"),
                    dmc.Space(h=20),
                    dmc.Group([
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    getIcon(icon="carbon:batch-job", size="lg"),
                                    dmc.Text("", id="job-info",weight=500)], position="left"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350),
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    getIcon(icon="iconoir:3d-three-pts-box", size="lg"),
                                    dmc.Text("", id="method-info", weight=500)], position="left"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350)], 
                        position="center", grow=1),
                    dmc.Space(h=15),
                    dmc.Group([
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    getIcon(icon="ic:round-loop", size="lg"),
                                    dmc.Text("", id="repeat-info",weight=500)], position="left"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350),
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    getIcon(icon="material-symbols:format-list-bulleted-add-rounded", size="lg"),
                                    dmc.Text("", id="variation-info", weight=500)], position="left"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350)], 
                        position="center", grow=1),
                    dmc.Space(h=20),
                    dmc.Group([
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    dmc.Text("Herunterladen als Text", weight=500),
                                    dmc.HoverCard([
                                        dmc.HoverCardTarget(dmc.ActionIcon(getIcon(icon="material-symbols:conversion-path", size="sm"), id="includeNormal-txt", color="gray", variant="transparent", n_clicks=0)),
                                        dmc.HoverCardDropdown(dmc.Text("Normalzustand testen?", size="sm"))], shadow="md")], 
                                    position="apart"), 
                                withBorder=True, inheritPadding=True, py="xs"),
                            dmc.CardSection(
                                dmc.List(
                                    id="txt-list", size="sm", spacing="sm", 
                                    children=[
                                        dmc.ListItem("nur relevante Codebausteine", icon=dmc.ThemeIcon(getIcon(icon="fluent:important-16-filled", size="sm"), radius="xl", color="gray.7", size=24)),
                                        dmc.ListItem("mit Normalzustand ohne Viapunkte", id="includeNormal-txt-list", icon=dmc.ThemeIcon(getIcon(icon="material-symbols:conversion-path", size="sm"), radius="xl", color="gray.7", size=24)),
                                        dmc.ListItem("als .txt-Datei", icon=dmc.ThemeIcon(getIcon(icon="lucide:file-text", size="sm"), radius="xl", color="gray.7", size=24)),
                                        dmc.ListItem("Code und Definitionen in einer Datei", icon=dmc.ThemeIcon(getIcon(icon="carbon:data-quality-definition", size="sm"), radius="xl", color="gray.7", size=24))]), 
                                withBorder=True, inheritPadding=True, py="xs"),
                            dmc.CardSection(
                                dmc.Group([
                                    dmc.Button("Herunteraden", id="download-txt", n_clicks=0)], 
                                    position="center"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350),
                        dmc.Card([
                            dmc.CardSection(
                                dmc.Group([
                                    dmc.Text("Modifiziertes Programm", weight=500),
                                    dmc.HoverCard([
                                        dmc.HoverCardTarget(dmc.ActionIcon(getIcon(icon="material-symbols:conversion-path", size="sm"), id="includeNormal-mod", color="gray", variant="transparent", n_clicks=0)),
                                        dmc.HoverCardDropdown(dmc.Text("Normalzustand testen?", size="sm"))], shadow="md")], 
                                    position="apart"), 
                                withBorder=True, inheritPadding=True, py="xs"),
                            dmc.CardSection(
                                dmc.List(
                                    id="txt-list", size="sm", spacing="sm", 
                                    children=[
                                        dmc.ListItem("neuen Job zum Programm hinzufügen", icon=dmc.ThemeIcon(getIcon(icon="streamline:programming-browser-add-app-code-apps-add-programming-window-plus", size="sm"), radius="xl", color="gray.7", size=24)),
                                        dmc.ListItem("mit Normalzustand ohne Viapunkte", id="includeNormal-mod-list", icon=dmc.ThemeIcon(getIcon(icon="material-symbols:conversion-path", size="sm"), radius="xl", color="gray.7", size=24)),
                                        dmc.ListItem("als .src-Datei und .dat-Datei", icon=dmc.ThemeIcon(getIcon(icon="ic:outline-file-copy", size="sm"), radius="xl", color="gray.7", size=24)),
                                        dmc.ListItem("Programm ist direkt ausführbar", icon=dmc.ThemeIcon(getIcon(icon="material-symbols:play-arrow-rounded", size="sm"), radius="xl", color="gray.7", size=24))]), 
                                withBorder=True, inheritPadding=True, py="xs"),
                            dmc.CardSection(
                                dmc.Group([
                                    dmc.Button("Herunteraden", id="download-mod", n_clicks=0), 
                                    dmc.Select(id="download-job-select", data=[], value=None, label=None, style={"width": 130}, rightSection=getIcon(icon="radix-icons:chevron-down", size="sm"))], 
                                    position="apart"), 
                                withBorder=True, inheritPadding=True, py="xs")], 
                            withBorder=True, shadow="sm", radius="md", miw=200, maw=350)], 
                        position="center", grow=1),
                    html.Div(id="notifications-container-download"),
                ]
            ),
        ]
    ),
    dmc.Space(h=20),
    dmc.Group([
        dmc.ActionIcon(getIcon(icon="material-symbols:arrow-back-ios-rounded", size="lg"), size="lg", variant="subtle", id="back-custom-icons", n_clicks=0), 
        dmc.ActionIcon(getIcon(icon="material-symbols:arrow-forward-ios-rounded", size="lg"), size="lg", variant="subtle", id="next-custom-icons", n_clicks=0)],
        position="apart", align="center"),
    html.Div(id="notifications-container-stepper"),
], align="stretch", justify="flex-start", maw=1000, m=20)


# ---------- Step 1 ---------- #

@robo_app.callback(
    output=dict(db = Output('dat-badge', 'color'),
                sb = Output('src-badge', 'color')),
    inputs=dict(content = Input('upload-data', 'contents')),
    state=dict(name = State('upload-data', 'filename'), 
               l_m = State('upload-data', 'last_modified'),
               dat_color = State('dat-badge', 'color'),
               src_color = State('src-badge', 'color')))
def updateStorage(content, name, l_m, dat_color, src_color):
    if content is None:
        raise PreventUpdate
    else:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)

        if '.dat' in name:
            data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep="no sep")
            data.to_csv("./dat" + ".csv")
            return dict(db="lime", sb=src_color)
        elif '.src' in name:
            data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep="no sep")
            data.to_csv("./src" + ".csv")
            return dict(db=dat_color, sb="lime")
        else:
            raise PreventUpdate


# ---------- Step 2 ---------- #

@robo_app.callback(
    output=dict(d = Output("job_input", "data"),
                jd = Output("download-job-select", "data"),
                jv = Output("download-job-select", "value")),
    inputs=dict(jobs = Input("jobs", "data")),
    state=dict(),
    prevent_initial_call=True,
)
def updateJobSelection(jobs):
    if jobs == None:
        raise PreventUpdate
    
    data = []
    for k in jobs.keys():
        data.append(k)

    return dict(d = data, jd = data, jv = None if len(data)==0 else data[-1])
    
@robo_app.callback(
    output=dict(c_t = Output("steps-line", "children"),
                a_t = Output("steps-line", "active"),
                c_c = Output("optimize-chips", "children"),
                v_c = Output("optimize-chips", "value"),
                fig = Output('path-plot', 'figure'),
                data = Output('jobs_tmp2', 'data'),
                gs = Output("GS-button", "variant"),
                code = Output("raw-code", "children")),
    inputs=dict(job = Input("job_input", "value"),
                search = Input("search-button", "n_clicks"),
                chips = Input("optimize-chips", "value"),
                gsClicks = Input("GS-button", "n_clicks"),
                theme = Input("mantineProvider", "theme")),
    state=dict(jobs = State("jobs", "data"),
               gsState = State("GS-button", "variant")),
)
def updatePath(job, search, chips, gsClicks, theme, jobs, gsState):
    # input checks
    templ = "plotly_dark" if theme["colorScheme"] == "dark" else "plotly_white"
    if job == None:
        return dict(c_t = [dmc.TimelineItem(title="Keine Job ausgewählt.")], a_t = None, 
                    c_c = [], v_c = [],
                    fig = go.Figure(go.Scatter3d(), layout=dict(template=templ)), data=jobs, gs=gsState, 
                    code="")
    elif len(jobs[job]["move"]) == 0:
        return dict(c_t = [dmc.TimelineItem(title="Keine Bewegungen gefunden.")], a_t = None, 
                    c_c = [], v_c = [],
                    fig = go.Figure(go.Scatter3d(), layout=dict(template=templ)), data=jobs, gs=gsState, 
                    code=getSrcCode(pd.read_csv("./src.csv", names=["lines"], engine="python"), job))
    

    # load job as DataFrame
    data = loadData(jobs[job], to_numeric=["X", "Y", "Z", "A", "B", "C"])

    if not "optimize" in list(data.columns): # init if not found
        data["optimize"] = np.zeros(len(data))
    
    # automatic search for to potential movements (logic: search fpr SPTP movements)
    if ctx.triggered_id == "search-button" and search != 0:
        data["optimize"] = np.zeros(len(data))
        for index, row in data.iterrows():
            if row["move"] == "SPTP":
                data.loc[index, "optimize"] = 1
    # highlight manually chosen movements
    elif len(chips) > 0:
        data["optimize"] = np.zeros(len(data))
        for chip in chips:
            data.loc[int(chip.split("_")[1]), "optimize"] = 1 


    # create separate DataFrame to show the user (changes to data_show will NOT be saved)
    data_show = copy.deepcopy(data)
    data_show["X"] = np.round(data_show["X"], decimals=1)
    data_show["Y"] = np.round(data_show["Y"], decimals=1)
    data_show["Z"] = np.round(data_show["Z"], decimals=1)
    data_show["A"] = np.round(data_show["A"], decimals=1)
    data_show["B"] = np.round(data_show["B"], decimals=1)
    data_show["C"] = np.round(data_show["C"], decimals=1)

    # filter Grundstellung if wished
    if ctx.triggered_id == "GS-button" and gsClicks != 0:
        if gsState == "outline":
            data_show = data_show[data_show["point"] != "grndstellung"]
            gsState = "subtle"
        else:
            gsState = "outline"
    elif gsState == "subtle":
        data_show = data_show[data_show["point"] != "grndstellung"]

    # Timeline
    timeline, chips, chip_values = createTimeline(data_show, getChips=True, theme=theme["colorScheme"])
          
    # Plot
    fig = createAnimatedPlot(data_show, theme=theme["colorScheme"])
    
    jobs[job] = data.to_dict()
    return dict(c_t = timeline, a_t = len(timeline), 
                c_c = chips, v_c = chip_values,
                fig = fig, data=jobs, gs=gsState,
                code=getSrcCode(pd.read_csv("./src.csv", names=["lines"], engine="python"), job))


# ---------- Step 3 ---------- #

@robo_app.callback(
    Output("via-modal", "opened"),
    Input("via-button", "n_clicks"),
    Input("close-button-modal", "n_clicks"),
    Input("submit-button-modal", "n_clicks"),
    State("via-modal", "opened"),
    prevent_initial_call=True,
)
def updateModal(nc1, nc2, nc3, opened):
    return not opened

@robo_app.callback(
    output=dict(d = Output("via-select-modal", "data")),
    inputs=dict(modjob = Input("modjob", "data")),
    state=dict(),
    prevent_initial_call=True,
)
def updateViaSelection(modjob):
    if modjob == None:
        raise PreventUpdate

    data = pd.DataFrame(modjob)

    return dict(d = data[(data["via"] == True) & (data["optimize"] == True)]["point"].values)

@robo_app.callback(
    output=dict(xvalue = Output("via-x-modal", "value"),
                xmin = Output("via-x-modal", "min"),
                xmax = Output("via-x-modal", "max"),
                yvalue = Output("via-y-modal", "value"),
                ymin = Output("via-y-modal", "min"),
                ymax = Output("via-y-modal", "max"),
                zvalue = Output("via-z-modal", "value"),
                zmin = Output("via-z-modal", "min"),
                zmax = Output("via-z-modal", "max"),
                avalue = Output("via-a-modal", "value"),
                amin = Output("via-a-modal", "min"),
                amax = Output("via-a-modal", "max"),
                bvalue = Output("via-b-modal", "value"),
                bmin = Output("via-b-modal", "min"),
                bmax = Output("via-b-modal", "max"),
                cvalue = Output("via-c-modal", "value"),
                cmin = Output("via-c-modal", "min"),
                cmax = Output("via-c-modal", "max")),
    inputs=dict(via = Input("via-select-modal", "value"),
                random = Input("via-random-modal", "n_clicks")),
    state=dict(modjob = State("modjob", "data")),
    prevent_initial_call=True,
)
def updateViaInputs(via, random, modjob):
    
    # prevent update if no via point selected
    if via == None:
        raise PreventUpdate
    
    # read modified job
    data = loadData(modjob, to_numeric=["X", "Y", "Z", "A", "B", "C"])

    # get point before and after the via point
    startp = data[len(data)-1:len(data)] if data[data["point"] == via].index[0] == 0 else data[data[data["point"] == via].index[0] - 1:data[data["point"] == via].index[0]]
    endp = data[data[data["point"] == via].index[0] + 1:data[data["point"] == via].index[0] + 2]

    # init X, Y, Z, A, B, C inputs in the form of (middle, min, max) (min > max is possible, because of 360°)
    X = (startp["X"].values[0] + (endp["X"].values[0]-startp["X"].values[0])/2 - (endp["X"].values[0]-startp["X"].values[0])/5,
         startp["X"].values[0] + (endp["X"].values[0]-startp["X"].values[0])/2 + (endp["X"].values[0]-startp["X"].values[0])/5)
    Y = (startp["Y"].values[0] + (endp["Y"].values[0]-startp["Y"].values[0])/2 - (endp["Y"].values[0]-startp["Y"].values[0])/5,
         startp["Y"].values[0] + (endp["Y"].values[0]-startp["Y"].values[0])/2 + (endp["Y"].values[0]-startp["Y"].values[0])/5)
    Z = (startp["Z"].values[0] + (endp["Z"].values[0]-startp["Z"].values[0])/2 - (endp["Z"].values[0]-startp["Z"].values[0])/5,
         startp["Z"].values[0] + (endp["Z"].values[0]-startp["Z"].values[0])/2 + (endp["Z"].values[0]-startp["Z"].values[0])/5)
    A = (startp["A"].values[0], endp["A"].values[0])
    B = (startp["B"].values[0], endp["B"].values[0])
    C = (startp["C"].values[0], endp["C"].values[0])

    # init first displayed values 
    if ctx.triggered_id == "via-random-modal" and random != 0:
        x, y, z, a, b, c = calcRandomPoint(startp, endp, 0)  
    else:
        x, y, z, a, b, c = X[0]+(X[1]-X[0])/2, Y[0]+(Y[1]-Y[0])/2, Z[0]+(Z[1]-Z[0])/2, A[0]+(A[1]-A[0])/2, B[0]+(B[1]-B[0])/2, C[0]+(C[1]-C[0])/2

    # return dysplayed values and borders for the input fields
    return dict(xvalue = x, xmin = X[0], xmax = X[1],
                yvalue = y, ymin = Y[0], ymax = Y[1],     
                zvalue = z, zmin = Z[0], zmax = Z[1],
                avalue = a, amin = A[0], amax = A[1],
                bvalue = b, bmin = B[0], bmax = B[1],
                cvalue = c, cmin = C[0], cmax = C[1])

@robo_app.callback(
    output=dict(fig = Output('move-plot', 'figure')),
    inputs=dict(viax = Input("via-x-modal", "value"),
                viay = Input("via-y-modal", "value"),
                viaz = Input("via-z-modal", "value"),
                theme = Input("mantineProvider", "theme")),
    state=dict(via = State("via-select-modal", "value"),
               modjob = State("modjob", "data"),
               xmin = State("via-x-modal", "min"),
               xmax = State("via-x-modal", "max"),
               ymin = State("via-y-modal", "min"),
               ymax = State("via-y-modal", "max"),
               zmin = State("via-z-modal", "min"),
               zmax = State("via-z-modal", "max")),
)
def updateMovePlot(viax, viay, viaz, theme, via, modjob, xmin, xmax, ymin, ymax, zmin, zmax):
    if via == None:
        if theme["colorScheme"] == "dark":
            return dict(fig = go.Figure(go.Scatter3d(), layout=dict(template="plotly_dark")))
        else:
            return dict(fig = go.Figure(go.Scatter3d(), layout=dict(template="plotly_white")))
 
    
    data = loadData(modjob, to_numeric=["X", "Y", "Z", "A", "B", "C"])

    startp = data[len(data)-1:len(data)] if data[data["point"] == via].index[0] == 0 else data[data[data["point"] == via].index[0] - 1:data[data["point"] == via].index[0]]
    endp = data[data[data["point"] == via].index[0] + 1:data[data["point"] == via].index[0] + 2]

    # Plot
    x = np.append(np.linspace(startp["X"].values[0], viax, 10, endpoint=True), np.linspace(viax, endp["X"].values[0], 10, endpoint=True)[1:])
    y = np.append(np.linspace(startp["Y"].values[0], viay, 10, endpoint=True), np.linspace(viay, endp["Y"].values[0], 10, endpoint=True)[1:])
    z = np.append(np.linspace(startp["Z"].values[0], viaz, 10, endpoint=True), np.linspace(viaz, endp["Z"].values[0], 10, endpoint=True)[1:])
    colors = np.array([plotly.colors.qualitative.Vivid[7]] * 9 + [plotly.colors.qualitative.Vivid[3]] + [plotly.colors.qualitative.Vivid[7]] * 9)

    # Create figure
    fig = go.Figure(go.Scatter3d(x=[], y=[], z=[], 
                                 mode="markers", marker=dict(color=[]), opacity=0.85)) # colorscale='Bluered_r'
    
    # Frames
    frames = [go.Frame(data = [go.Scatter3d(x=x[:k], y=y[:k], z=z[:k], marker=dict(color=colors[:k]))],
                       traces = [0], name = f'frame{k}') for k  in  range(len(x)+1)]

    fig.update(frames=frames)

    sliders = [{"pad": {"b": 10, "t": 60}, "len": 0.9, "x": 0.1, "y": 0, 
                "steps": [{"args": [[f.name], frame_args(0)],
                           "label": str(k),
                           "method": "animate",} for k, f in enumerate(fig.frames)]}]

    fig.update_layout(updatemenus = [{"buttons":[{"args": [None, frame_args(50)], "label": "Start", "method": "animate",},
                                                 {"args": [[None], frame_args(0)], "label": "Pause", "method": "animate",}],
                                                  "direction": "left", "pad": {"r": 10, "t": 70}, "type": "buttons", "x": 0.1, "y": 0, "bgcolor": "yellowgreen", "showactive": False}],
                      sliders=sliders)

    fig.update_layout(scene = dict(xaxis=dict(range=[min(x), max(x)], autorange=False),
                                   yaxis=dict(range=[min(y), max(y)], autorange=False),
                                   zaxis=dict(range=[min(z), max(z)], autorange=False)))

    if theme["colorScheme"] == "dark":
        fig.update_layout(sliders=sliders, width=800, height=800, template="plotly_dark")
    else:
        fig.update_layout(sliders=sliders, width=800, height=800, template="plotly_white")

    fig.add_trace(go.Scatter3d(x=[startp["X"].values[0], endp["X"].values[0]], y=[startp["Y"].values[0], endp["Y"].values[0]], z=[startp["Z"].values[0], endp["Z"].values[0]], line=dict(color='black', width=2), marker=dict(size=0), name="lin. Interp."))
    
    x_ = np.linspace(xmin, xmax, 10, endpoint=True)
    y_ = np.linspace(ymin, ymax, 10, endpoint=True)
    z_ = np.linspace(zmin, zmax, 10, endpoint=True)
    X, Y, Z = np.meshgrid(x_, y_, z_, indexing='ij')
    values = np.ones(X.shape)
    fig.add_trace(go.Isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=values.flatten(), opacity=0.6, showscale=False, colorscale="Greens"))

    return dict(fig = fig)

@robo_app.callback(
    output=dict(c_t = Output("modsteps-line", "children"),
                a_t = Output("modsteps-line", "active"),
                fig = Output('modpath-plot', 'figure'),
                code = Output("modraw-code", "children"),
                data = Output('modjob_tmp2', 'data'),
                via_data = Output("vias", "data")),
    inputs=dict(trial_method = Input("trial_method", "value"),
                submitVia = Input("submit-button-modal", "n_clicks"),
                randomVia = Input("via-random", "n_clicks"),
                nloops = Input("nloops", "value"),
                nvias = Input("nvias", "value"),
                theme = Input("mantineProvider", "theme")),
    state=dict(modjob = State("modjob", "data"),
               viap = State("via-select-modal", "value"),
               viax = State("via-x-modal", "value"),
               viay = State("via-y-modal", "value"),
               viaz = State("via-z-modal", "value"),
               viaa = State("via-a-modal", "value"),
               viab = State("via-b-modal", "value"),
               viac = State("via-c-modal", "value"),
               viaData = State("vias", "data"),
               job = State("job_input", "value"),
               theme = State("mantineProvider", "theme")),
)
def updateModPath(trial_method, submitVia, randomVia, nloops, nvias, theme, modjob, viap, viax, viay, viaz, viaa, viab, viac, viaData, job):
    
    # prevent calculations if user inputs missing
    templ = "plotly_dark" if theme["colorScheme"] == "dark" else "plotly_white"
    if modjob == None or trial_method == None:
        return dict(c_t = [dmc.TimelineItem(title="Keine Versuchsmethode gefunden.")], a_t = None, 
                    fig = go.Figure(go.Scatter3d(), layout=dict(template=templ)), data=None, via_data=viaData,
                    code="")
    elif len(modjob["move"]) == 0:
        return dict(c_t = [dmc.TimelineItem(title="Keine Bewegungen gefunden.")], a_t = None, 
                    fig = go.Figure(go.Scatter3d(), layout=dict(template=templ)), data=modjob, via_data=viaData,
                    code="")

    # read modified job
    data = loadData(modjob, to_numeric=["X", "Y", "Z", "A", "B", "C"])

    # get current via assignments
    viaData = {} if viaData == None else viaData

    # assign koordinates to specific via point
    if ctx.triggered_id == "submit-button-modal" and submitVia != 0:
        if viap in viaData.keys():
            viaData[viap]["X"][0] = np.round(viax, 3)
            viaData[viap]["Y"][0] = np.round(viay, 3)
            viaData[viap]["Z"][0] = np.round(viaz, 3)
            viaData[viap]["A"][0] = np.round(viaa, 3)
            viaData[viap]["B"][0] = np.round(viab, 3)
            viaData[viap]["C"][0] = np.round(viac, 3)
        else:
            viaData[viap] = {"X": [viax], "Y": [viay], "Z": [viaz], "A": [viaa], "B": [viab], "C": [viac]}

    # random assignment to all via points
    if ctx.triggered_id == "via-random" and randomVia != 0:
        for index, row in data[(data["via"] == True) & (data["optimize"] == 1)].iterrows():
            startp = data[len(data)-1:len(data)] if index == 0 else data[index-1:index]
            endp = data[index+1:index+2]

            x, y, z, a, b, c = calcRandomPoint(startp, endp, nvias)

            viaData[row["point"]] = {"X": x, "Y": y, "Z": z, "A": a, "B": b, "C": c}
            
    # change number of via points 
    if ctx.triggered_id == "nvias" and nvias != None:
        for index, row in data[(data["via"] == True) & (data["optimize"] == 1)].iterrows():
            startp = data[len(data)-1:len(data)] if index == 0 else data[index-1:index]
            endp = data[index+1:index+2]

            x, y, z, a, b, c = calcRandomPoint(startp, endp, nvias)
                
            viaData[row["point"]] = {"X": x, "Y": y, "Z": z, "A": a, "B": b, "C": c}
                
    # create trial set
    data_show = createTrialData(data, viaData, trialMethod=trial_method, nloops=nloops, nvias=nvias)

    data_show["X"] = np.round(np.array(data_show["X"]).astype(float), decimals=1)
    data_show["Y"] = np.round(np.array(data_show["Y"]).astype(float), decimals=1)
    data_show["Z"] = np.round(np.array(data_show["Z"]).astype(float), decimals=1)
    data_show["A"] = np.round(np.array(data_show["A"]).astype(float), decimals=1)
    data_show["B"] = np.round(np.array(data_show["B"]).astype(float), decimals=1)
    data_show["C"] = np.round(np.array(data_show["C"]).astype(float), decimals=1)

    # Timeline
    timeline = createTimeline(data_show, theme=theme["colorScheme"])

    # Plot
    fig = createAnimatedPlot(data_show, theme=theme["colorScheme"])
    
    # Code
    code = getSrcCode(pd.read_csv("./src.csv", names=["lines"], engine="python"), job, getString=False)
    modcode = pd.DataFrame({"lines": []})
    
    if trial_method == "einfaches Viapunkt-Set" or trial_method == "verschienene Viapunkte":
        akt_line, via_i = 0, 0
        for index, row in data_show.iterrows():
            if row["move"] == "FOR":
                if via_i == nvias:
                    via_i = 0
                akt_line = 0
                while "*******" in code.loc[akt_line, "lines"] or "Job" in code.loc[akt_line, "lines"] or ("JJ" in code.loc[akt_line, "lines"] and "OO" in code.loc[akt_line, "lines"] and "BB" in code.loc[akt_line, "lines"]):
                    akt_line += 1
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["FOR i = 1 TO " + str(row["point"]) + " STEP 1"]})])

            elif row["via"] == True:
                while not data_show.loc[index+1,"move"] in code.loc[akt_line, "lines"]:
                    modcode = pd.concat([modcode, code[akt_line:akt_line+1]])
                    akt_line += 1
                if trial_method == "einfaches Viapunkt-Set":
                    modcode = pd.concat([modcode, pd.DataFrame({"lines": ["PTP " + str(row["point"]) + " C_PTP"]})])
                else:
                    modcode = pd.concat([modcode, pd.DataFrame({"lines": ["PTP " + str(row["point"]) + str(via_i) + " C_PTP"]})])

            elif row["move"] in ["PTP", "SPTP", "LIN", "SLIN"]:
                while not data_show.loc[index,"move"] in code.loc[akt_line, "lines"]:
                    modcode = pd.concat([modcode, code[akt_line:akt_line+1]])
                    akt_line += 1
                if "FOLD" in code.loc[akt_line, "lines"]:
                    while not "ENDFOLD" in code.loc[akt_line, "lines"]:
                        modcode = pd.concat([modcode, code[akt_line:akt_line+1]])
                        akt_line += 1
                    modcode = pd.concat([modcode, code[akt_line:akt_line+1]])
                    akt_line += 1

            elif row["move"] == "ENDFOR":
                while akt_line <= len(code):
                    modcode = pd.concat([modcode, code[akt_line:akt_line+1]])
                    akt_line += 1
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["ENDFOR"]})])
                via_i += 1

            elif row["move"] == "WAIT":
                while akt_line <= len(code):
                    modcode = pd.concat([modcode, code[akt_line:akt_line+1]])
                    akt_line += 1
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["WAIT SEC " + row["point"].split(" ")[1]]})])

    elif trial_method == "DoE Plan pro Bewegung":
        via_i = 0
        for index, row in data_show.iterrows():
            if row["move"] == "FOR":
                if via_i == nvias:
                    via_i = 0
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["FOR i = 1 TO " + str(row["point"]) + " STEP 1"]})])
            elif row["via"] == True:
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["PTP " + str(row["point"]) + str(via_i) + " C_PTP"]})])
            elif row["move"] in ["PTP", "SPTP", "LIN", "SLIN"]:
                modcode = pd.concat([modcode, 
                                     pd.DataFrame({"lines": ["PTP " + str(row["point"])]}),
                                     pd.DataFrame({"lines": ["WAIT SEC 0.2"]})])
            elif row["move"] == "ENDFOR":
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["ENDFOR"]})])
                via_i += 1
            elif row["move"] == "WAIT":
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["WAIT SEC " + row["point"].split(" ")[1]]})])        
        

    code_str = ""
    for index, row in modcode.iterrows():
        if "ENDFOR" in row["lines"]:
            code_str = code_str + str(row["lines"]) + "\n"
            code_str = code_str + " \n"
        elif "FOR" in row["lines"]:
            code_str = code_str + " \n"
            code_str = code_str + str(row["lines"]) + "\n"
        else:
            code_str = code_str + str(row["lines"]) + "\n"
        

    modjob = data.to_dict()
    return dict(c_t = timeline, a_t = len(timeline), 
                fig = fig, data=modjob, via_data=viaData,
                code=code_str)


# -------- Step Done --------- #

@robo_app.callback(
    output=dict(ntl = Output("includeNormal-txt-list", "children"),
                nt = Output("includeNormal-txt", "children")),
    inputs=dict(normalButton = Input("includeNormal-txt", "n_clicks")),
    state=dict(normalText = State("includeNormal-txt-list", "children")),
    prevent_initial_call=False,
)
def updateCard(normalButton, normalText):
    if ctx.triggered_id == "includeNormal-txt" and normalButton != 0:
        if "mit" in normalText:
            return dict(ntl="ohne Normalzustand ohne Viapunkte", nt=getIcon(icon="material-symbols:conversion-path-off-rounded", size="sm"))
        return dict(ntl="mit Normalzustand ohne Viapunkte", nt=getIcon(icon="material-symbols:conversion-path", size="sm"))
    raise PreventUpdate

@robo_app.callback(
    output=dict(nml = Output("includeNormal-mod-list", "children"),
                nm = Output("includeNormal-mod", "children")),
    inputs=dict(modButton = Input("includeNormal-mod", "n_clicks")),
    state=dict(modText = State("includeNormal-mod-list", "children")),
    prevent_initial_call=False,
)
def updateCard(modButton, modText):
    if ctx.triggered_id == "includeNormal-mod" and modButton != 0:
        if "mit" in modText:
            return dict(nml="ohne Normalzustand ohne Viapunkte", nm=getIcon(icon="material-symbols:conversion-path-off-rounded", size="sm"))
        return dict(nml="mit Normalzustand ohne Viapunkte", nm=getIcon(icon="material-symbols:conversion-path", size="sm"))
    raise PreventUpdate

@robo_app.callback(
    output=dict(n = Output("notifications-container-download", "children"),),
    inputs=dict(download_txt = Input("download-txt", "n_clicks"),
                download_mod = Input("download-mod", "n_clicks")),
    state=dict(trial_method = State("trial_method", "value"),
               nloops = State("nloops", "value"),
               nvias = State("nvias", "value"),
               modjob = State("modjob", "data"),
               viaData = State("vias", "data"),
               job = State("job_input", "value"),
               normalText_txt = State("includeNormal-txt-list", "children"),
               normalText_mod = State("includeNormal-mod-list", "children"),
               saveJob = State("download-job-select", "value")),
    prevent_initial_call=False,
)
def download(download_txt, download_mod, trial_method, nloops, nvias, modjob, viaData, job, normalText_txt, normalText_mod, saveJob): 

    # prevent downoad if no button pressed
    if download_txt == 0 and download_mod == 0:
        raise PreventUpdate

    # read modified job
    data = loadData(modjob, to_numeric=["X", "Y", "Z", "A", "B", "C"])

    # create trail DateFrame
    includeNormal = ("mit" in normalText_txt and ctx.triggered_id == "includeNormal-txt") or ("mit" in normalText_mod and ctx.triggered_id == "includeNormal-mod")
    data_show = createTrialData(data, viaData, trialMethod=trial_method, includeNormal=includeNormal, nloops=nloops, nvias=nvias)

    # Code
    code = getSrcCode(pd.read_csv("./src.csv", names=["lines"], engine="python"), job, getString=False)
    modcode = pd.DataFrame({"lines": []})
    
    if trial_method == "einfaches Viapunkt-Set" or trial_method == "verschienene Viapunkte":
        akt_line, via_i = 0, 0
        for index, row in data_show.iterrows():
            if row["move"] == "FOR":
                if via_i == nvias:
                    via_i = 0
                akt_line = 0
                while "*******" in code.loc[akt_line, "lines"] or "Job" in code.loc[akt_line, "lines"] or ("JJ" in code.loc[akt_line, "lines"] and "OO" in code.loc[akt_line, "lines"] and "BB" in code.loc[akt_line, "lines"]):
                    akt_line += 1
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["FOR i = 1 TO " + str(row["point"]) + " STEP 1"]})])

            elif row["via"] == True:
                while not data_show.loc[index+1,"move"] in code.loc[akt_line, "lines"]:
                    modcode = pd.concat([modcode, code[akt_line:akt_line+1]])
                    akt_line += 1
                if trial_method == "einfaches Viapunkt-Set":
                    modcode = pd.concat([modcode, pd.DataFrame({"lines": ["PTP " + str(row["point"]) + " C_PTP"]})])
                else:
                    modcode = pd.concat([modcode, pd.DataFrame({"lines": ["PTP " + str(row["point"]) + str(via_i) + " C_PTP"]})])

            elif row["move"] in ["PTP", "SPTP", "LIN", "SLIN"]:
                while not data_show.loc[index,"move"] in code.loc[akt_line, "lines"]:
                    modcode = pd.concat([modcode, code[akt_line:akt_line+1]])
                    akt_line += 1
                if "FOLD" in code.loc[akt_line, "lines"]:
                    while not "ENDFOLD" in code.loc[akt_line, "lines"]:
                        modcode = pd.concat([modcode, code[akt_line:akt_line+1]])
                        akt_line += 1
                    modcode = pd.concat([modcode, code[akt_line:akt_line+1]])
                    akt_line += 1

            elif row["move"] == "ENDFOR":
                while akt_line <= len(code):
                    modcode = pd.concat([modcode, code[akt_line:akt_line+1]])
                    akt_line += 1
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["ENDFOR"]})])
                via_i += 1

            elif row["move"] == "WAIT":
                while akt_line <= len(code):
                    modcode = pd.concat([modcode, code[akt_line:akt_line+1]])
                    akt_line += 1
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["WAIT SEC " + row["point"].split(" ")[1]]})])

    elif trial_method == "DoE Plan pro Bewegung":
        via_i = 0
        for index, row in data_show.iterrows():
            if row["move"] == "FOR":
                if via_i == nvias:
                    via_i = 0
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["FOR i = 1 TO " + str(row["point"]) + " STEP 1"]})])
            elif row["via"] == True:
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["PTP " + str(row["point"]) + str(via_i) + " C_PTP"]})])
            elif row["move"] in ["PTP", "SPTP", "LIN", "SLIN"]:
                modcode = pd.concat([modcode, 
                                     pd.DataFrame({"lines": ["PTP " + str(row["point"])]}),
                                     pd.DataFrame({"lines": ["WAIT SEC 0.2"]})])
            elif row["move"] == "ENDFOR":
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["ENDFOR"]})])
                via_i += 1
            elif row["move"] == "WAIT":
                modcode = pd.concat([modcode, pd.DataFrame({"lines": ["WAIT SEC " + row["point"].split(" ")[1]]})])        
        
    code_str = "int i \n"
    for index, row in modcode.iterrows():
        if "ENDFOR" in row["lines"]:
            code_str = code_str + str(row["lines"]) + "\n"
            code_str = code_str + " \n"
        elif "FOR" in row["lines"]:
            code_str = code_str + " \n"
            code_str = code_str + str(row["lines"]) + "\n"
        else:
            code_str = code_str + str(row["lines"]) + "\n"
    
    dat_str, via_i, vias = "", 0, []
    for index, row in data_show[(data_show["via"] == True) & (data_show["optimize"] == 1)].iterrows():
        if str(row["point"]) in vias:
            via_i += 1
            vias = []
        vias.append(str(row["point"]))

        dat_str = dat_str + "DECL POS " + str(row["point"]) + str(via_i) + "={X " + str(row["X"]) + ",Y " + str(row["Y"]) + ",Z " + str(row["Z"]) + ",A " + str(row["A"]) + ",B " + str(row["B"]) + ",C " + str(row["C"]) + ",S " + str(row["S"]) + ",T " + str(row["T"]) + "}" + "\n"
    dat_str = dat_str + "\n"

    if ctx.triggered_id == "download-txt" and download_txt != 0:
        with open('Code.txt', 'w') as f:
            f.write("; modifizierter src-Code:" + "\n")
            f.write(code_str)
            f.write('\n')
            f.write("; neue Punkte als dat-Code:" + "\n")
            f.write(dat_str)
        
        return dict(n = dmc.Notification(title="Download abgeschlossen", action="show", id="notification",
                                         message="Der Code wurde als Code.txt im App-Verzeichnis gespeichert.",
                                         icon=getIcon(icon="akar-icons:circle-check", size="sm"), color="lime"))

    elif ctx.triggered_id == "download-mod" and download_mod != 0:
        src = pd.read_csv("./src.csv", names=["lines"], engine="python")
        dat = pd.read_csv("./dat.csv", names=["lines"], engine="python")

        srcStr, src_jobstart = "", src[src.index >= src[src["lines"] == "GLOBAL DEF UP_" + saveJob].index[0] + 9]
        for index, row in src[src.index <= src[src["lines"] == "GLOBAL DEF UP_" + saveJob].index[0] + 9].iterrows():
            srcStr = srcStr + str(row["lines"]) + "\n"
        srcStr = srcStr + code_str
        for index, row in src_jobstart[src_jobstart.index >= src_jobstart[src_jobstart["lines"].str.contains("END ")].index[0]].iterrows():
            srcStr = srcStr + str(row["lines"]) + "\n"
        
        datStr = ""
        if len(dat[dat["lines"].str.contains("Job " + saveJob.split("_")[1][:2])]) > 0:
            for index, row in dat[dat.index <= dat[dat["lines"].str.contains("Job " + saveJob.split("_")[1][:2])].index[0] + 2].iterrows():
                datStr = datStr + str(row["lines"]) + "\n"
            datStr = datStr + dat_str
            for index, row in dat[dat.index > dat[dat["lines"].str.contains("Job " + saveJob.split("_")[1][:2])].index[0] + 2].iterrows():
                datStr = datStr + str(row["lines"]) + "\n"
        else:
            for index, row in dat.iterrows():
                datStr = datStr + str(row["lines"]) + "\n"
            datStr = datStr + dat_str

        with open('mod.src', 'w') as f:
            f.write(srcStr)

        with open('mod.dat', 'w') as f:
            f.write(datStr)

        return dict(n = dmc.Notification(title="Download abgeschlossen", action="show", id="notification",
                                         message="Die modifizierten mod.src und mod.dat wurden im App-Verzeichnis gespeichert.",
                                         icon=getIcon(icon="akar-icons:circle-check", size="sm"), color="lime"))    
    
    raise PreventUpdate


# ---------- Stepper --------- #

@robo_app.callback(
    output=dict(s = Output("stepper-custom-icons", "active"),
                j = Output("jobs_tmp1", "data"),
                mj = Output("modjob_tmp1", "data"),
                ji = Output("job-info", "children"),
                mi = Output("method-info", "children"),
                ri = Output("repeat-info", "children"),
                vi = Output("variation-info", "children"),
                n = Output("notifications-container-stepper", "children")),
    inputs=dict(back = Input("back-custom-icons", "n_clicks"),
                next = Input("next-custom-icons", "n_clicks")),
    state=dict(current = State("stepper-custom-icons", "active"),
               dat_c = State('dat-badge', 'color'),
               src_c = State('src-badge', 'color'),
               jobs = State("jobs", "data"),
               viaData = State("vias", "data"),
               chosen_job = State("job_input", "value"),
               trial_method = State("trial_method", "value"),
               nloops = State("nloops", "value"),
               nvias = State("nvias", "value")),
    prevent_initial_call=True,
)
def updateStepper(back, next, current, dat_c, src_c, jobs, viaData, chosen_job, trial_method, nloops, nvias):
    
    # prevent accidential callbacks
    if (ctx.triggered_id == "back-custom-icons" and back == 0) or (ctx.triggered_id == "next-custom-icons" and next == 0):
        raise PreventUpdate

    # return inits
    modified_job = None
    job_info, method_info, repeat_info, variation_info = "", "", "", ""
    
    # get current step
    step = current if current is not None else 0


    # -- STEP 0 --

    if step == 0 and src_c == "red": # no src file uploaded
        return dict(s = step, j = None, mj = None, ji="", mi="", ri="", vi="", 
                    n=dmc.Notification(title="Keine .src Datei gefunden", action="show", id="notification",
                                       message="Bitte lade eine .src Datei hoch.",
                                       icon=getIcon(icon="mdi:bell-alert-outline", size="sm"), color="red")) 
    if step == 0 and dat_c == "red": # no dat file uploaded
        return dict(s = step, j = None, mj = None, ji="", mi="", ri="", vi="", 
                    n=dmc.Notification(title="Keine .dat Datei gefunden", action="show", id="notification",
                                       message="Bitte lade eine .dat Datei hoch.",
                                       icon=getIcon(icon="mdi:bell-alert-outline", size="sm"), color="red")) 
    
    if step == 0 and jobs == None: # initialize job if not already done
        
        # read .dat-lines as dataframe and just keep POS definitions and Grundstellung
        dat = pd.read_csv("./dat.csv", names=["lines"], engine="python")
        dat = dat[dat["lines"].str.contains("DECL")]
        dat = dat[dat["lines"].str.contains("POS") | dat["lines"].str.contains("HILFS_GRDST")]
        
        # search point names in lines 
        points = []
        for index, row in dat.iterrows():
            if "POS" in row["lines"]:
                points.append(row["lines"].split("POS")[1].split("=")[0].lower().replace("x", "").replace(" ", ""))
            elif "HILFS_GRDST" in row["lines"]:
                points.append("grndstellung")
            else:
                raise "konnte Punkt (" + row["lines"] + ") nicht identifizieren"

        # define new columns in dataframe: pointname and positons 
        dat["point"] = points
        dat['X'] = pd.to_numeric(dat['lines'].map(lambda line: line.split("{")[1].split("X ")[1].split(",")[0]))
        dat['Y'] = pd.to_numeric(dat['lines'].map(lambda line: line.split("{")[1].split("Y ")[1].split(",")[0]))
        dat['Z'] = pd.to_numeric(dat['lines'].map(lambda line: line.split("{")[1].split("Z ")[1].split(",")[0]))
        dat['A'] = pd.to_numeric(dat['lines'].map(lambda line: line.split("{")[1].split("A ")[1].split(",")[0]))
        dat['B'] = pd.to_numeric(dat['lines'].map(lambda line: line.split("{")[1].split("B ")[1].split(",")[0]))
        dat['C'] = pd.to_numeric(dat['lines'].map(lambda line: line.split("{")[1].split("C ")[1].split(",")[0].replace("}", "")))
        dat['S'] = pd.to_numeric(dat['lines'].map(lambda line: 0 if not "S " in line else line.split("{")[1].split("S ")[1].split(",")[0]))
        dat['T'] = pd.to_numeric(dat['lines'].map(lambda line: 0 if not "T " in line else line.split("{")[1].split("T ")[1].split(",")[0]))
        dat.drop(["lines"], axis=1, inplace=True)

        # read .src-lines as dataframe
        src = pd.read_csv("./src.csv", names=["lines"], engine="python")

        # search jobs in file
        jobs = []
        for index, row in src.iterrows():
            if "UP" in row["lines"][:2] and "Job" in row["lines"]:
                jobs.append(row["lines"][3:])

        # get program movements for all jobs and save in dict
        program_job = {}
        for job in jobs:
            
            # get lines in job
            code = src[src.index > src[src["lines"] == "GLOBAL DEF UP_" + job].index[0]]
            code = code[code.index < code[code["lines"].str.contains("END ")].index[0]]

            # filter important lines with informations obout the movements
            code = code[code["lines"].str.contains("PTP") | code["lines"].str.contains("LIN") | code["lines"].str.contains("FOR")]
            code.reset_index(drop=True, inplace=True)
            code.drop(np.array(code[code["lines"].str.contains("FOLD")].index)+1, inplace=True)
            code['lines'] = code['lines'].map(lambda line: line.replace(";FOLD ", ""))
            code['move'] = code['lines'].map(lambda line: line.split(" ")[0])
            code['via'] = code['lines'].map(lambda line: "C_PTP" in line)

            # filter point names
            points = []
            for index, row in code.iterrows():
                if row["move"] == "FOR":
                    points.append(int(row["lines"].split("TO ")[1].split(" ")[0]))
                elif row["move"] == "ENDFOR":
                    points.append(None)
                else:
                    points.append(row["lines"].split(" ")[1].split(" ")[0].lower())
            code["point"] = points
            code.drop(["lines"], axis=1, inplace=True)
    
            # merge code with dat informations and save to dict
            code = code.merge(dat, how="left", on="point")
            program_job[job] = code.to_dict()
        jobs = program_job


    # -- STEP 1 --

    if step == 1 and chosen_job == None: # no job chosen
        return dict(s = step, j = None, mj = None, ji="", mi="", ri="", vi="", 
                    n=dmc.Notification(title="Kein Job ausgewählt", action="show", id="notification",
                                       message="Bitte wähle einen Job aus, der optimiert werden soll.",
                                       icon=getIcon(icon="mdi:bell-alert-outline", size="sm"), color="red")) 
    
    if step == 1: # add via points to chosen job
        
        # get unmodified job from storage
        data = pd.DataFrame(jobs[chosen_job])
        data.reset_index(drop=True, inplace=True)
        
        # modifi job with via points at the chosen movements
        i, last_index = 0, 0
        data_mod = data[:0]
        for index, row in data[data["optimize"] == 1].iterrows():
            data_mod = pd.concat([data_mod, data[last_index:index]])
            data_mod = pd.concat([data_mod, pd.DataFrame({"move": ["PTP"], "via": [True], "point": ["via" + str(i)], "X": [None], "Y": [None], "Z": [None], "A": [None], "B": [None], "C": [None], "S": [data.loc[index-1 if index!=0 else index+1,"S"]], "T": [data.loc[index-1 if index!=0 else index+1,"T"]], "optimize": [1]})])
            last_index = index
            i += 1
        data_mod = pd.concat([data_mod, data[last_index:]])
        data_mod.reset_index(drop=True, inplace=True)

        # save modified job
        modified_job = data_mod.to_dict()


    # -- STEP 2 --

    if step == 2 and trial_method == None: # no trial method chosen
        return dict(s = step, j = None, mj = None, ji="", mi="", ri="", vi="", 
                    n=dmc.Notification(title="Keine Versuchmethode ausgewählt", action="show", id="notification",
                                       message="Bitte wähle eine der drei Versuchsmethoden zur Gestaltung des Programms aus.",
                                       icon=getIcon(icon="mdi:bell-alert-outline", size="sm"), color="red")) 
    if step == 2 and (len(viaData) < nvias or len(viaData[list(viaData.keys())[0]]["X"]) != nvias): # unassigned viapoints found
        return dict(s = step, j = None, mj = None, ji="", mi="", ri="", vi="", 
                    n=dmc.Notification(title="Viapunkte nicht definiert", action="show", id="notification",
                                       message="Es wurden Viapunkt(e) gefunden, deren Lage nicht eindeutig definiert ist.",
                                       icon=getIcon(icon="mdi:bell-alert-outline", size="sm"), color="red")) 

    if step == 2: # save informations for summary
        job_info = chosen_job
        method_info = trial_method
        repeat_info = str(nloops) + " Wiederholungen pro Viapunkt"
        variation_info = str(nvias) + " Variationen pro Viapunkt"


    # -- CHANGE STEP --
    
    min_step, max_step = 0, 3
    if ctx.triggered_id == "back-custom-icons":
        step = step - 1 if step > min_step else step
    else:
        step = step + 1 if step < max_step else step

    return dict(s = step, j = jobs, mj = modified_job, ji=job_info, mi=method_info, ri=repeat_info, vi=variation_info, n=None)

    
# -------- Side Menu -------- #

@robo_app.callback(output=dict(pc = Output("page-content", "children"),
                          ha = Output("home-link", "active"), 
                          ea = Output("edit-link", "active")), 
              inputs=dict(pathname = Input("url", "pathname")))
def renderPageContent(pathname):
    if pathname == "/":
        return dict(pc=home_layout, ha=True, ea=False)
    elif pathname == "/edit":
        return dict(pc=edit_layout, ha=False, ea=True)

    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


if __name__ == "__main__":
    robo_app.run_server()



""" 
TO DOs: 
    - Leerzeichen in mod.src und mod.dat erhalten
    - Mehr Kommentare in mod Programm
    - Code Kommenieren

    - DOE Zusammenfassung Downloaden

"""
