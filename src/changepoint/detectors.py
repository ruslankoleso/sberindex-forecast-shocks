"""Онлайн-детекторы точек структурных изменений.

Каждый детектор получает ряд x (лог-отклонение МО от «типичного МО» в тот же месяц)
и обрабатывает его последовательно: решение в месяце t принимается только по x[:t+1].
Возвращает score[t] — силу сигнала в месяце t; тревога, когда score[t] > порога.
Порог подбирается одинаково для всех детекторов (одинаковая доля ложных тревог),
поэтому детекторы сравниваются честно.

warmup — сколько первых месяцев детектор только «смотрит» и не тревожит.
"""
import numpy as np


def _robust_scale(x):
    """Масштаб шума по приращениям (устойчиво к выбросам)."""
    d = np.diff(x)
    mad = np.median(np.abs(d - np.median(d))) * 1.4826 if len(d) else np.nan
    return max(mad / np.sqrt(2), 1e-3)


def cusum(x, warmup=8, drift=0.5):
    """Двусторонний CUSUM: накопленные отклонения от среднего «обычного» периода.
    Среднее и масштаб оцениваются по первым warmup месяцам; drift — допуск в сигмах."""
    x = np.asarray(x, float)
    mu, sd = x[:warmup].mean(), _robust_scale(x[:warmup])
    sp = sn = 0.0
    score = np.zeros(len(x))
    for t in range(warmup, len(x)):
        z = (x[t] - mu) / sd
        sp, sn = max(0.0, sp + z - drift), max(0.0, sn - z - drift)
        score[t] = max(sp, sn)
    return score


def page_hinkley(x, warmup=8, delta=0.5):
    """Тест Пейджа–Хинкли: отклонение накопленной суммы от её экстремума (в сигмах)."""
    x = np.asarray(x, float)
    sd = _robust_scale(x[:warmup])
    score = np.zeros(len(x))
    for t in range(warmup, len(x)):
        z = (x[:t + 1] - np.mean(x[:t + 1])) / sd
        m = np.cumsum(z - delta * np.sign(z))
        up = m[-1] - m.min()
        down = m.max() - m[-1]
        score[t] = max(up, down) / np.sqrt(t + 1)
    return score


def bocpd(x, warmup=8, hazard=1 / 24, window=3):
    """Байесовское онлайн-обнаружение (Adams & MacKay, 2007), гауссова модель с
    неизвестным средним; вычисления в логарифмах (без переполнения).
    score[t] — вероятность того, что текущий режим начался в последние `window` месяцев."""
    from scipy.special import logsumexp
    x = np.asarray(x, float)
    sd = _robust_scale(x[:warmup])
    mu0, k0 = x[:warmup].mean(), 1.0
    logR = np.array([0.0])
    mus, ks = np.array([mu0]), np.array([k0])
    lh, l1h = np.log(hazard), np.log(1 - hazard)
    score = np.zeros(len(x))
    for t in range(len(x)):
        var = sd ** 2 * (1 + 1 / ks)
        lpred = -0.5 * (x[t] - mus) ** 2 / var - 0.5 * np.log(2 * np.pi * var)
        growth = logR + lpred + l1h
        cp = logsumexp(logR + lpred + lh)
        logR = np.append(cp, growth)
        logR -= logsumexp(logR)
        mus = np.append(mu0, (ks * mus + x[t]) / (ks + 1))
        ks = np.append(k0, ks + 1)
        if t >= warmup:
            score[t] = np.exp(logsumexp(logR[1:window + 1]))
    return score


def pelt_online(x, warmup=8, recent=3, penalty=2.0):
    """PELT (ruptures) в расширяющемся окне: в месяце t сегментируем x[:t+1].
    Если последний найденный перелом b лежит в последних `recent` месяцах,
    score[t] — выигрыш от разбиения в b: насколько уменьшается сумма квадратов
    отклонений (в единицах дисперсии шума) по сравнению с одним общим средним
    на отрезке от предыдущего перелома до t. Иначе 0."""
    import ruptures as rpt
    x = np.asarray(x, float)
    sd = _robust_scale(x[:warmup])
    z = x / sd
    sse = lambda v: float(((v - v.mean()) ** 2).sum()) if len(v) else 0.0
    score = np.zeros(len(x))
    for t in range(warmup, len(x)):
        seg = z[:t + 1]
        bkps = rpt.Pelt(model="l2", min_size=2, jump=1).fit(seg.reshape(-1, 1)).predict(pen=penalty)[:-1]
        if bkps and bkps[-1] >= t + 1 - recent:
            b = bkps[-1]
            a0 = bkps[-2] if len(bkps) > 1 else 0
            score[t] = sse(seg[a0:]) - sse(seg[a0:b]) - sse(seg[b:])
    return score


def forecast_residual(x, warmup=8, window=6, consec=2):
    """Детектор по ошибке прогноза: прогноз месяца t — среднее предыдущих `window` месяцев;
    score[t] — минимальная из последних `consec` стандартизованных ошибок (по модулю,
    одного знака), т. е. факт устойчиво уходит от прогноза, а не разово."""
    x = np.asarray(x, float)
    sd = _robust_scale(x[:warmup])
    e = np.full(len(x), np.nan)
    for t in range(warmup, len(x)):
        e[t] = (x[t] - x[max(0, t - window):t].mean()) / sd
    score = np.zeros(len(x))
    for t in range(warmup + consec - 1, len(x)):
        last = e[t - consec + 1:t + 1]
        if np.all(np.sign(last) == np.sign(last[0])):
            score[t] = np.abs(last).min()
    return score


DETECTORS = {
    "cusum": cusum,
    "page_hinkley": page_hinkley,
    "bocpd": bocpd,
    "pelt": pelt_online,
    "forecast_residual": forecast_residual,
}


def confirmed(det, consec=2):
    """Обёртка «подтверждение»: сигнал месяца t = минимум сигнала за последние `consec`
    месяцев. Разовый выброс даёт один всплеск и отсеивается; ценой — +1 мес. задержки."""
    def run(x, warmup=8, **kw):
        s = det(x, warmup=warmup, **kw)
        out = np.zeros_like(s)
        for t in range(warmup + consec - 1, len(s)):
            out[t] = s[t - consec + 1:t + 1].min()
        return out
    run.__name__ = f"{det.__name__}_confirmed"
    return run


DETECTORS.update({
    "bocpd_confirmed": confirmed(bocpd),
    "pelt_confirmed": confirmed(pelt_online),
})
