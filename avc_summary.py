from pandas_profiling import ProfileReport
import pandas as pd

# Get the previous overall data

df = pd.read_csv('avc_dataML.csv',sep=";")

# Get the previous overall data stats

report = ProfileReport(df, minimal=True).to_html()

