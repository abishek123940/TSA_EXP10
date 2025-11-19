# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 12.11.2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, difflib, warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

def clean_num(x):
    try: return float(str(x).replace(',', '').replace('%', '').strip())
    except: return np.nan

def find_country(name, cutoff=0.5):
    c_list = df['Country'].dropna().astype(str).tolist()
    m = difflib.get_close_matches(name, c_list, n=1, cutoff=cutoff)
    return m[0] if m else None

df = pd.read_csv("world_population.csv")
df.columns = ['Rank','CCA3','Country','Capital','Continent',
              'Pop_2022','Pop_2015','Pop_2010','Pop_2000',
              'Pop_1990','Pop_1980','Pop_1970','Area_km2',
              'Density_per_km2','Growth_Rate','World_Pop_Percent','Extra']
for c in ['Pop_1970','Pop_1980','Pop_1990','Pop_2000','Pop_2010','Pop_2015','Pop_2022']:
    df[c] = df[c].apply(clean_num)
df = df[df['Country'].notna()].reset_index(drop=True)

user = input("Enter country: ").strip()
match = find_country(user) or find_country(user, 0.3)
if not match: raise SystemExit("Country not found.")
print("Using:", match)

r = df[df['Country'] == match].iloc[0]
years = np.array([1970,1980,1990,2000,2010,2015,2022])
vals = np.array([r[f'Pop_{y}'] for y in years], float)
mask = ~np.isnan(vals)
ts = pd.Series(vals[mask], index=pd.PeriodIndex(years[mask], freq='Y'))

plt.plot(ts.index.to_timestamp(), ts, 'o-', label="Population")
plt.title(f"{match} - Population Trend")
plt.xlabel("Year"); plt.ylabel("Population"); plt.grid(); plt.show()

if len(ts) >= 3:
    stat, p = adfuller(ts)[0:2]
    print(f"ADF p-value={p:.4f}", "(Stationary)" if p<=0.05 else "(Non-stationary)")
    lags = max(1, min(len(ts)//2 - 1, 6))
    plot_acf(ts, lags=lags); plt.show()
    plot_pacf(ts, lags=lags, method='ywm'); plt.show()

train, test = ts[:int(len(ts)*0.8)], ts[int(len(ts)*0.8):]
model = SARIMAX(train, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

if not test.empty:
    pred = model.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
    plt.plot(train.index.to_timestamp(), train, 'bo-', label='Train')
    plt.plot(test.index.to_timestamp(), test, 'go-', label='Test')
    plt.plot(pred.index.to_timestamp(), pred, 'rx-', label='Predicted')
    plt.legend(); plt.grid(); plt.show()
    print(f"RMSE: {np.sqrt(mean_squared_error(test, pred)):.2f}")

last, end = ts.index[-1].year, 2050
steps = end - last
fc = model.get_forecast(steps)
fc_mean, fc_ci = fc.predicted_mean, fc.conf_int()
future = pd.period_range(last+1, periods=steps, freq='Y')
fc_mean.index = fc_ci.index = future

plt.plot(ts.index.to_timestamp(), ts, 'bo-', label='Historical')
plt.plot(fc_mean.index.to_timestamp(), fc_mean, 'o-', label='Forecast', color='orange')
plt.fill_between(fc_ci.index.to_timestamp(), fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.25, color='orange')
plt.title(f"{match} - Forecast to {end}")
plt.legend(); plt.grid(); plt.show()

```

### OUTPUT:
<img width="567" height="453" alt="download" src="https://github.com/user-attachments/assets/a49b60ca-48e3-420b-8205-4ac09188a598" />

<img width="568" height="433" alt="download" src="https://github.com/user-attachments/assets/961f86e1-b40a-4abb-879c-e7817d621555" />

<img width="568" height="433" alt="download" src="https://github.com/user-attachments/assets/a2b97439-aa2c-460f-8bc4-388024a9eef0" />

<img width="547" height="428" alt="download" src="https://github.com/user-attachments/assets/f6849fde-1409-4136-ae32-1e86ac27ff0d" />

<img width="547" height="433" alt="download" src="https://github.com/user-attachments/assets/863bcde0-0cdf-4baf-b13f-2d080d6b26e5" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
