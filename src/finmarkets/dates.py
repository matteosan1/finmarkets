from datetime import date
from dateutil.relativedelta import relativedelta

def TimeInterval(interval):
    """
    Callable that translate strings into relativedeltas

    Params:
    -------
    interval: str
        the string representing the time interval
    """
    tag = interval[-1].lower()
    value = int(interval[:-1])
    if tag == "d":
        return relativedelta(days=value)
    elif tag == "m":
        return relativedelta(months=value)
    elif tag == "y":
        return relativedelta(years=value)
    else:
        raise ValueError(f"Unable to convert {interval}, probably wrong units.")
    
def generate_dates(start_date, end_date, frequency="1y"):
    """
    Computes a set of dates given starting date and length in months.

    Params:
    -------
    start_date: datetime.date
        the start date of the set of dates
    end_date: str or datetime.date
        maturity that defines the length of the list of dates
    frequency: str
        frequency of the list of dates, by default is 12 months
    """
    if isinstance(end_date, str):
        end_date = start_date + TimeInterval(end_date)
    d = start_date
    dates = [start_date]
    while True:
        d += TimeInterval(frequency)
        if d < end_date:
            dates.append(d)
        else:
            dates.append(end_date)
            break
    return dates

