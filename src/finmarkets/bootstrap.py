from scipy.optimize import newton

class Bootstrap:
    def __init__(self, obs_date, objects):
        self.obs_date = obs_date
        self.objects = objects

    def objective_function(self, x, i, xs, curve, kwargs):
        pillars = [self.obs_date] + [self.objects[j].payment_dates[-1] for j in range(i+1)]
        c = curve(self.obs_date, pillars, [1] + xs + [x])
        return self.objects[i].npv(c, **kwargs)

    def run(self, curve, guess=1.0, kwargs={}):
        x = []    
        for i in range(len(self.objects)):
            res = newton(self.objective_function, guess, args=(i, x, curve, kwargs))
            x.append(res)
        return x