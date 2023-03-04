import pandas as pd
import numpy as np
import holidays


df = pd.read_csv("demand_2022_utc.csv")

df["Time (UTC)"] = pd.to_datetime(df["Time (UTC)"].str[0:16], utc=True, dayfirst=True)

df = df.rename(columns={"Time (UTC)": "time", "Actual Total Load [MW] - BZN|AT": "demand"})
df = df.filter(["time", "demand"])

at_holidays = holidays.AT()
# call it once
"2022-01-01" in at_holidays
df["holiday"] = df.time.dt.date.isin(at_holidays)
df["weekday"] = df.time.dt.weekday

first_monday = (df.weekday == 0).idxmax()
last_sunday = (df.weekday == 6)[::-1].idxmax()


df = df.loc[first_monday:last_sunday]
df = df.reset_index(drop=True)

def day_type(data: pd.Series):
    if data.holiday:
        return 6
    return data.weekday

df["day_type"] = df.apply(lambda x: day_type(x), axis=1)

df = pd.concat([df, pd.get_dummies(df.day_type)], axis=1)

max_demand = df.demand.max()
# df["demand_scale"] = df.demand / max_demand

weeks = [df.loc[(df.index // 168) == i] for i in range(len(df) // 168)]


MAX_SCALE = 5000

def avg_demand(data: pd.Series):
    return data.demand.mean()

def scaled_demand(data: pd.Series):
    return (data.demand - avg_demand(data)) / MAX_SCALE

def get_day(week, i):
    day = week.iloc[(i*24):(i+1)*24].copy()
    day["average_demand"] = avg_demand(day)
    day["average_demand_scale"] = avg_demand(day) / max_demand
    day.demand = scaled_demand(day)
    return day

def day2input(day):
    return np.concatenate((np.random.normal(size=2), day.iloc[0][[0,1,2,3,4,5,6,"average_demand_scale"]].values)).astype(float)

def day2output(day):
    return day.demand.values.astype(float)


day2input(get_day(weeks[0], 1))
day2output(get_day(weeks[0], 1))

def get_sample(weeks, w, d):
    return (
        day2input(get_day(weeks[w], d)),
        day2output(get_day(weeks[w], d)),
    )

def random_sample(weeks):
    return get_sample(weeks, np.random.randint(len(weeks)), np.random.randint(7))


samples = [
    get_sample(weeks, w, d)
    for w in range(len(weeks))
    for d in range(7)
    for i in range(1)
]
