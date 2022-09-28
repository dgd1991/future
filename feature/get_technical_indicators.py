def get_kdj(x, data):
    if x.name == 0:
        return [data.loc[x.name]['rsv'], data.loc[x.name]['rsv'], data.loc[x.name]['rsv']]
    else:
        k_value = 2/3 * data.loc[x.name-1]['k_value'] + 1/3 * data.loc[x.name]['rsv']
        d_value = data.loc[x.name-1]['d_value'] + 1/3 * k_value
        j_value = 3 * k_value - 2 * d_value
        return [k_value, d_value, j_value]
def get_macd(x, data):
    if x.name == 0:
        ema_12 = data.loc[x.name]['close']
        ema_26 = data.loc[x.name]['close']
        dif = ema_12 - ema_26
        dea = dif
        dea = (dif-dea)*2
        return [dif, dea, dea]
    else:
        ema_12 = 11/13 * ema_12 + 2/13 * data.loc[x.name]['close']
        ema_26 = data.loc[x.name]['close']
        dif = ema_12 - ema_26
        dea = dif
        dea = (dif - dea) * 2

        k_value = 2/3 * data.loc[x.name-1]['k_value'] + 1/3 * data.loc[x.name]['rsv']
        d_value = data.loc[x.name-1]['d_value'] + 1/3 * k_value
        j_value = 3 * k_value - 2 * d_value
        return [k_value, d_value, j_value]