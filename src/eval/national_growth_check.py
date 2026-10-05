"""Проверка: как прогнозировать годовой рост России — «рост сохранится», ETS или Chronos-2.
Оценка на национальном ряде, origin 2021-12…2026-07. Запуск: python -m src.eval.national_growth_check"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from src.forecast.national import _load_L, forecast_national
L=_load_L()
from chronos import BaseChronosPipeline
pipe=BaseChronosPipeline.from_pretrained('amazon/chronos-2',device_map='cpu')
rows=[]
for t0 in L.index[(L.index>='2021-12-01')]:
    hist=L[L.index<=t0]
    for meth in ['persist','ets','chronos2']:
        if meth=='persist':
            g=hist.iloc[-1]-hist.iloc[-13]; fut=None
        else:
            fut=forecast_national(hist,12,meth,pipe)
        for h in (1,3,6,12):
            t=t0+pd.DateOffset(months=h)
            if t not in L.index: continue
            true_g=L[t]-L[t-pd.DateOffset(years=1)]
            pred_g=g if meth=='persist' else fut[h-1]-L[t-pd.DateOffset(years=1)]
            rows.append((meth,h,t0,abs(np.exp(pred_g)-np.exp(true_g))*100))
r=pd.DataFrame(rows,columns=['метод','h','origin','ошибка_пп'])
print('точек прогноза:',r.origin.nunique(), r.origin.min().date(), r.origin.max().date())
print(r.pivot_table(index='метод',columns='h',values='ошибка_пп',aggfunc='mean').round(2).to_string())
r['год']=r.origin.dt.year
print(r[r.h==12].pivot_table(index='метод',columns='год',values='ошибка_пп',aggfunc='mean').round(1).to_string())
