import math
def float2Bucket(raw, base, floor, ceil, bucket_Size):
    if not raw or math.isnan(raw):
        return None
    else:
        raw = float(raw)
        if raw < floor:
            return int(bucket_Size + 1)
        elif raw > ceil:
            return int(bucket_Size + 2)
        else:
            return int(math.floor(raw * base) % bucket_Size)


def bignumber2Bucket(raw, log, bucket_Size, is_abs):
    if not raw or math.isnan(raw):
        return None
    else:
        raw = float(raw)
        if is_abs:
            bucket_Size = bucket_Size / 2
            if raw < 0:
                raw = abs(raw)
                if raw < 1:
                    return bucket_Size * 2 + 1
                else:
                    return int(math.log(raw, log) % bucket_Size)
            else:
                if raw < 1:
                    return bucket_Size * 2 + 2
                else:
                    return int(math.log(raw, log) % bucket_Size) + bucket_Size
        else:
            if raw < 1:
                return bucket_Size + 1
            else:
                return int(math.log(raw, log) % bucket_Size)


def get_date_quarter(year, month, last_quarter):
    if last_quarter:
        if month in ('01', '02', '03'):
            return str(year) + str(2)
        elif month in ('04', '05', '06'):
            return str(year) + str(3)
        elif month in ('07', '08', '09'):
            return str(year) + str(4)
        elif month in ('10', '11', '12'):
            return str(int(year) + 1) + str(1)
        else:
            return None
    else:
        if month in ('01', '02', '03'):
            return str(year) + str(1)
        elif month in ('04', '05', '06'):
            return str(year) + str(2)
        elif month in ('07', '08', '09'):
            return str(year) + str(3)
        elif month in ('10', '11', '12'):
            return str(year) + str(4)
        else:
            return None
def get_date_previous_quarter(year, month, is_previous):
    if is_previous:
        if month in ('01', '02', '03'):
            return str(int(year) - 1), str(4)
        elif month in ('04', '05', '06'):
            return str(year), str(1)
        elif month in ('07', '08', '09'):
            return str(year), str(2)
        elif month in ('10', '11', '12'):
            return str(year), str(3)
        else:
            return None, None
    else:
        if month in ('01', '02', '03'):
            return str(year), str(1)
        elif month in ('04', '05', '06'):
            return str(year), str(2)
        elif month in ('07', '08', '09'):
            return str(year), str(3)
        elif month in ('10', '11', '12'):
            return str(year), str(4)
        else:
            return None, None

def get_normal_label(diff):
    if diff >= -0.5 and diff < 0.5:
        return 0.5
    elif diff >= 0.5 and diff < 2:
        return 0.65
    elif diff >= 2 and diff < 4:
        return 0.75
    elif diff >= 4 and diff < 6:
        return 0.85
    elif diff >= 6 and diff < 8:
        return 0.95
    elif diff >= 8:
        return 1.0
    elif diff >= -2 and diff < -0.5:
        return 0.35
    elif diff >= -4 and diff < -2:
        return 0.25
    elif diff >= -6 and diff < -4:
        return 0.15
    elif diff >= -8 and diff < -6:
        return 0.05
    elif diff < -8:
        return 0.0

def get_classification_label(diff):
    if diff > 0:
        return 1.0
    elif diff <= 0:
	    return 0.0
	    return -1
    else:
        pass