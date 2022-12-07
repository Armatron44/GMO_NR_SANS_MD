#!/usr/bin/python3
"""Calculates SLD from mass density profiles"""
from functools import reduce

import pandas as pd
import plotly.graph_objects as go

# prefix for new file_names
name = "gmo_1_water_1"
# partial names of the files to load
types = [
    "dod_c_sld",
    "dod_h_sld",
    "gmo_c_sld",
    "gmo_h_sld",
    "gmo_o_sld",
    "wat_h_sld",
    "wat_o_sld",
    "wat_d_sld"
]

folder = "average_05"

# iterative load of all files of form {typ}.dat
files_to_read = (
    pd.read_csv(
        f"{folder}/{typ}.dat",
        header=0,
        names=["z", f"{typ}"],
        delimiter=r"\s+",
    )
    for typ in types
)

# merging files, using 'z' as the merging paramenter
df = reduce(
    lambda x, y: pd.merge(x, y, on="z", how="outer"),
    [x for x in files_to_read],
)
# re-sorting values in orderof ascending 'z' and replacing NaN with 0
df = df.sort_values("z", axis=0).reset_index(drop=True).fillna(0)
# 'normalizing' 'z' so it starts at 0
df["z"] = df["z"] - df["z"][0]

# deep copy of df as to not modify the original
h_df = df.copy()
# calculate half the count of 'z'
half_length = int(h_df["z"].count() / 2)
# redifining h_df['z'] to be drop the last half
h_df["z"] = h_df["z"][:half_length]
# iteratively fold each type
for typ in types:
    h_df[typ] = (
        h_df[typ][:half_length]
        + h_df[typ][half_length:].iloc[::-1].reset_index(drop=True)
    ) / 2
# that leaves half the rows with NaN. Removing those rows
h_df = h_df.dropna(axis=0, how="all")

# assigning alias for certain columns / sums of columns
gmo = df["gmo_c_sld"] + df["gmo_h_sld"] + df["gmo_o_sld"]
dod = df["dod_c_sld"] + df["dod_h_sld"]
wat = df["wat_h_sld"] + df["wat_o_sld"]
wat_d = df["wat_d_sld"] + df["wat_o_sld"]
total = gmo + dod
total_d = gmo + dod + wat_d

h_df["gmo"] = h_df["gmo_c_sld"] + h_df["gmo_h_sld"] + h_df["gmo_o_sld"]
h_df["dodecane"] = h_df["dod_c_sld"] + h_df["dod_h_sld"]
h_df["water"] = h_df["wat_h_sld"] + h_df["wat_o_sld"]
h_df["water_d"] = h_df["wat_d_sld"] + h_df["wat_o_sld"]
h_df["total"] = h_df["gmo"] + h_df["dodecane"]
h_df["total_d"] = h_df["gmo"] + h_df["dodecane"] + h_df["water_d"]

# write half_df to file
h_df.to_csv(
    f"{name}_sld.csv",
    columns=["z", "gmo", "dodecane", "water", "water_d", "total"],
    index=False,
)

# Create original SLD figure object
fig_sld = go.Figure()

# Original SLD graph
fig_sld.add_trace(go.Scatter(x=df["z"], y=total, name="total"))

fig_sld.add_trace(go.Scatter(x=df["z"], y=gmo, name="gmo"))

fig_sld.add_trace(go.Scatter(x=df["z"], y=dod, name="dodecane"))

fig_sld.add_trace(go.Scatter(x=df["z"], y=wat, name="water"))

# options
fig_sld.update_layout(title="GMO neutron SLD profile", showlegend=True)

# Axes options
fig_sld.update_xaxes(title_text="<i>z</i> / Å")
fig_sld.update_yaxes(title_text="SLD")

# Save as html
fig_sld.write_html(f"{name}_total.html")

# show on browser
fig_sld.show()


# Folded SLD
# create figure object
fig_half_sld = go.Figure()

fig_half_sld.add_trace(go.Scatter(x=h_df["z"], y=h_df["total"], name="total"))

fig_half_sld.add_trace(go.Scatter(x=h_df["z"], y=h_df["gmo"], name="gmo"))

fig_half_sld.add_trace(
    go.Scatter(x=h_df["z"], y=h_df["dodecane"], name="dodecane")
)

fig_half_sld.add_trace(go.Scatter(x=h_df["z"], y=h_df["water"], name="water"))

# options
fig_half_sld.update_layout(
    title="GMO neutron SLD profile",
    #  plot_bgcolor='rgb(230,230,230)',
    showlegend=True,
)

# Axes options
fig_half_sld.update_xaxes(title_text="<i>z</i> / Å")
fig_half_sld.update_yaxes(title_text="SLD")

# Save as html
fig_half_sld.write_html(f"{name}_sld.html")

# show in browser
fig_half_sld.show()
