import pandas as pd
from tools.Tools import Tools
tools = Tools()
print(tools.get_recent_month_date(str(2009)+'-01'+'-01', -15))