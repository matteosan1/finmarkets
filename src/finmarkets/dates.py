from datetime import date
from dateutil.relativedelta import relativedelta
from enum import Enum

def timeinterval(interval):
    tag = interval[-1].lower()
    value = int(interval[:-1])
    if tag == "d":
        return relativedelta(days=value)
    elif tag == "m":
        return relativedelta(months=value)
    elif tag == "y":
        return relativedelta(years=value)
    
def generate_dates(start_date, end_date, tenor="1y"):
    """
    Computes a set of dates given starting date and length in months.

    Params:
    -------
    start_date: datetime.date
        the start date of the set of dates
    end_date: str or datetime.date
        maturity that defines the length of the list of dates
    tenor: str
        tenor of the list of dates, by default is 12 months
    """
    if isinstance(end_date, str):
        end_date = start_date + timeinterval(end_date)
    d = start_date
    dates = [start_date]
    while True:
        d += timeinterval(tenor)
        if d < end_date:
            dates.append(d)
        else:
            dates.append(end_date)
            break
    return dates

