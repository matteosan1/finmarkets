from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from enum import Enum

IntervalType = Enum("IntervalType", {"Annual":"1Y", "Semiannual":"6M", "Quarterly":"3M", "Daily":"1D"})

class Interval:
    def __init__(self, interval):
        if type(interval) == IntervalType:
            interval = interval.value
        self.tag = interval[-1].lower()
        if self.tag not in ['y', 'd', 'm']:
            raise ValueError(f"Wrong time unit {self.tag}")
        self.value = int(interval[:-1])
        
    def to_days(self):
        if self.tag == "y":
            val = self.value * 360
        elif self.tag == "m":
            val = self.value*30
        elif self.tag == "w":
            val = self.value*7
        elif self.tag == "d":
            val = self.value
            
    def tau(self):
        if self.tag == "y":
            return self.value
        elif self.tag == "m":
            return self.value/12
        elif self.tag == "d":
            return self.value/365
    
    def add_to(self, other):
        if not isinstance(other, date):
            raise TypeError("Can only add Interval to datetime.date")
        if self.tag == "d":
            new_date = other + relativedelta(days=self.value)
        elif self.tag == "m":
            new_date = other + relativedelta(months=self.value)
        elif self.tag == "y":
            new_date = other + relativedelta(years=self.value)
        return new_date

def generate_dates(start_date, end_date, tenor=Interval(IntervalType.Annual)):
    """
    Computes a set of dates given starting date and length in months.

    Params:
    -------
    start_date: datetime.date
        the start date of the set of dates
    end_date: Interval or datetime.date
        maturity that defines the length of the list of dates
    tenor: Interval
        tenor of the list of dates, by default is 12 months
    """
    if isinstance(end_date, Interval):
        end_date = end_date.add_to(start_date)
    d = start_date
    dates = [start_date]
    while True:
        d = tenor.add_to(d)
        if d < end_date:
            dates.append(d)
        else:
            dates.append(end_date)
            break
    return dates

if __name__ == "__main__":
    print (generate_dates(date(2023, 10, 20), Interval("1y")))
