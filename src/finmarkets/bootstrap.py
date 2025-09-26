from .global_const import GlobalConst

from scipy.optimize import newton

class Bootstrap:
    def __init__(self, objects):
        self.objects = objects

    def get_pillars(self, idx=-1):
        if idx == -1:
            idx = len(self.objects)-1
        return [self.objects[j].payment_dates[-1] for j in range(idx+1)]

    def objective_function(self, x, i, xs, curve, kwargs):
        c = curve(self.get_pillars(i), xs + [x])
        return self.objects[i].npv(c, **kwargs)

    def run(self, curve, guess=1.0, kwargs={}):
        x = []    
        for i in range(len(self.objects)):
            res = newton(self.objective_function, guess, args=(i, x, curve, kwargs))
            x.append(res)
        return curve(self.get_pillars(), x)