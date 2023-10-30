from datetime import date
from dateutil.relativedelta import relativedelta

def maturity_from_str(maturity, unit="m"):
    """
    Utility to convert time intervals to integers into days, months (default) or years. 
    The interval has the following format "XXy" with XX the value and y the units (y, Y, m, M, d, D).

    Params:
    -------
    maturity: str
        the string to be converted
    unit: str
        time unit of the output, default value is month
    """
    tag = maturity[-1].lower()
    maturity = float(maturity[:-1])
    if tag == "y":
        maturity *= 12
    elif tag == "d":
        maturity /= 30.417
    elif tag != "m":
        raise ValueError(f"Unrecognized label {tag}")

    unit = unit.lower()
    if unit == "y":
        maturity /= 12
    elif unit == "d":
        maturity *= 30.417
    elif unit != "m":
        raise ValueError(f"Unrecognized output unit {unit}")
    
    return maturity

def dates_diff(d1, d2, unit="m"):
    if d1 > d2:
        raise ValueError("d1 must be lower than d2.")
    return (d2 - d1).days/30.417

def generate_dates(start_date, maturity, tenor="1y", unit="m"):
    """
    Computes a set of dates given starting date and length in months.

    Params:
    -------
    start_date: datetime.date
        the start date of the set of dates
    maturity: str
        maturity that defines the length of the list of dates
    tenor: str
        tenor of the list of dates, by default is 12 months
    unit: str
        unit to compute shifts
    """
    shift = {"years":0, "months":0, "days":0}
    final_shift = {"years":0, "months":0, "days":0}
    maturity = int(round(maturity_from_str(maturity, unit), 0))
    tenor = int(round(maturity_from_str(tenor, unit), 0))

    dates = []
    
    for d in range(0, maturity, tenor):
        if unit.lower() == "d":
            dates.append(start_date + relativedelta(days=d))
        elif unit.lower() == "m":
            dates.append(start_date + relativedelta(months=d))
        elif unit.lower() == "y":
            dates.append(start_date + relativedelta(years=d))


    if unit.lower() == "d":
        dates.append(start_date + relativedelta(days=maturity))
    elif unit.lower() == "m":
        dates.append(start_date + relativedelta(months=maturity))
    elif unit.lower() == "y":
        dates.append(start_date + relativedelta(years=maturity))
    return dates

if __name__ == "__main__":
    print (generate_dates(date(2023, 10, 20), "1y", "1m", "d"))
