import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

NUMBER = 6


class TaskFunc:
    def __init__(self, n):
        self.alpha = 2 + 0.1 * n
        c1 = 1
        c2 = 1
        self.f = lambda x, y: y + self.alpha * x * (1 - x) + 2 * self.alpha + 2
        self.solution = lambda x: self.alpha * x ** 2 - self.alpha * x + c1 * np.e ** (-x) + c2 * np.e ** x - 2
        self.pre_fire_func = lambda y, dy: dy - np.e + 1 / np.e - self.alpha

    def get_alphas_for_running(self):
        return 0, -self.alpha

    def get_betas_for_running(self):
        return 0, np.e - 1 / np.e  + self.alpha

    def get_funcs_for_running(self):
        p = lambda x: 1
        q = lambda x: self.alpha * x * (1 - x) + 2 * self.alpha + 2
        return p, q


class CalcMethod:
    def __init__(self, tf):
        self.tf = tf

    def ret_plot(self, nc):
        return None, None


class exact_solution(CalcMethod):
    def __init__(self, tf):
        super().__init__(tf)
        self.name = 'Exact solution'
        self.color = 'black'

    def ret_plot(self, nc):
        x_res = np.linspace(0, 1, 1000)
        y_res = [self.tf.solution(x) for x in x_res]
        return x_res, y_res


class FrontEnd:
    def __init__(self, m):
        self.methods = m
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        tbbox = plt.axes([0.1, 0.05, 0.8, 0.075])
        self.tb = TextBox(tbbox, 'Шаги', initial='25')
        self.tb.on_submit(self.redraw)

    def show(self):
        self.redraw('25')
        plt.show()

    def redraw(self, steps_raw):
        # plt.xkcd()
        self.ax.cla()
        self.ax.set_xlim([0, 1])

        self.ax.set_ylim([-1, 1])
        try:
            steps = int(steps_raw)
        except ValueError:
            return
        for m in self.methods:
            s = m.ret_plot(steps)
            self.ax.plot(s[0], s[1], color=m.color, label=m.name)
            plt.draw()
        self.ax.legend()


class ShootingMethod(CalcMethod):
    def __init__(self, tf):
        super().__init__(tf)
        self.f = tf.f
        self.pre_fire_func = tf.pre_fire_func
        self.start = 0
        self.end = 1
        self.alpha = tf.alpha
        self.eps = 1e-10

    def solve(self, nc, y0, dy0):
        x_res = np.linspace(self.start, self.end, nc + 1)
        y_res = [y0]
        dy_res = [dy0]
        for n in range(len(x_res) - 1):
            h = x_res[n + 1] - x_res[n]
            ynext, dynext = self.main_calc(self.f, n, h, x_res[n], y_res[n], dy_res[n])
            y_res.append(ynext)
            dy_res.append(dynext)
        return x_res, y_res, dy_res

    def main_calc(self, f, n, h, x, y, dy):
        return None, None

    def dichotomy(self, a, b, f):
        fa = f(a)
        fb = f(b)
        if fa == 0:
            return a
        if fb == 0:
            return b
        if fa * fb > 0:
            raise ValueError()
        while abs(a - b) > self.eps:
            c = (a + b) / 2
            fc = f(c)
            if fc == 0:
                return c
            if fc * fa < 0:
                b = c
                fb = fc
                continue
            if fc * fb < 0:
                a = c
                fa = fc
                continue
            raise ValueError()
        return (a + b) / 2

    def pre_fire(self, m, nc):
        x, y, dy = self.solve(nc, m, -self.alpha)
        return self.pre_fire_func(y[nc], dy[nc])

    def _fire(self, nc):
        mbig = self.dichotomy(-1, 1, lambda m: self.pre_fire(m, nc))
        x_res, y_res, dy_res = self.solve(nc, mbig, -self.alpha)
        return x_res, y_res, dy_res

    def ret_plot(self, nc):
        return self._fire(nc)


class runge_kutte(ShootingMethod):
    def __init__(self, tf):
        super().__init__(tf)
        self.name = 'Runge Kutte'
        self.color = '#f18f01'

    def main_calc(self, f, n, h, xn, yn, dyn):
        k1y = h * dyn
        k1dy = h * f(xn, yn)

        k2y = h * (dyn + k1dy / 2)
        k2dy = h * f(xn + h / 2, yn + k1y / 2)  # , dyn + k1dy / 2)

        k3y = h * (dyn + k2dy / 2)
        k3dy = h * f(xn + h / 2, yn + k2y / 2)  # , dyn + k2dy / 2)

        k4y = h * (dyn + k3dy)
        k4dy = h * f(xn + h, yn + k3y)  # , dyn + k3dy)

        ynext = yn + (k1y + 2 * (k2y + k3y) + k4y) / 6
        dynext = dyn + (k1dy + 2 * (k2dy + k3dy) + k4dy) / 6
        return ynext, dynext


class euler_recalc(ShootingMethod):
    def __init__(self, tf):
        super().__init__(tf)
        self.color = '#048ba8'
        self.name = 'Euler with calc'

    def main_calc(self, f, n, h, xn, yn, dyn):
        dynext = dyn + h / 2 * (f(xn, yn) + f(xn + h, yn + h * dyn))
        ynext = yn + h / 2 * (dyn + dyn + h * f(xn, yn))
        return ynext, dynext


class euler_explicit(ShootingMethod):
    def __init__(self, tf):
        super().__init__(tf)
        self.color = '#de4057'
        self.name = 'Euler explicit'

    def main_calc(self, f, n, h, xn, yn, dyn):
        dynext = dyn + h * f(xn, yn)
        ynext = yn + h * dyn
        return ynext, dynext


class RunningAlgo(CalcMethod):
    def __init__(self, tf):
        super().__init__(tf)
        self.alpha0, self.alpha1 = tf.get_alphas_for_running()
        self.beta0, self.beta1 = tf.get_betas_for_running()
        self.p, self.q = tf.get_funcs_for_running()

    def ret_plot(self, nc):
        pass


class Approx1(RunningAlgo):
    def __init__(self, tf):
        super().__init__(tf)
        self.color = '#99c24d'
        self.name = 'Approximate One'
        pass

    def ret_plot(self, nc):
        x_res = np.linspace(0, 1, nc+1)

        h = (x_res[nc] - x_res[0]) / nc

        lambdas = np.zeros((nc + 2,), dtype=np.float64)
        lambdas[1] = 1 / (1 + h * self.alpha0)

        mus = np.zeros((nc + 2,), dtype=np.float64)
        mus[1] = -h * self.alpha1 / (1 + h * self.alpha0)

        for n in range(1, nc):
            x_n = x_res[n]
            a_n = 2 + self.p(x_n) * h ** 2
            b_n = self.q(x_n) * h ** 2
            lambdas[n + 1] = -1 / (lambdas[n] - a_n)
            mus[n + 1] = (b_n - mus[n]) / (lambdas[n] - a_n)

        y_res = np.zeros(x_res.shape, dtype=np.float64)


        lambda_n = 1 - h * self.beta0
        mu_n = -h * self.beta1

        y_res[nc] = -(mus[nc] - mu_n) / (lambdas[nc] - lambda_n)

        for n in range(nc, 0, -1):
            y_res[n - 1] = lambdas[n] * y_res[n] + mus[n]
        return x_res, y_res


class Approx2(RunningAlgo):
    def __init__(self, tf):
        super().__init__(tf)
        self.color = 'violet'
        self.name = 'Approximate Two'
        pass

    def ret_plot(self, nc):
        x_res = np.linspace(0, 1, nc+1)
        h = (x_res[nc] - x_res[0]) / nc

        k = 2 * h * self.alpha0 + self.p(x_res[0]) * h ** 2 + 2

        lambdas = np.zeros((nc + 2,), dtype=np.float64)
        lambdas[1] = 2 / k

        mus = np.zeros((nc + 2,), dtype=np.float64)
        mus[1] = -(2 * h * self.alpha1 + self.q(x_res[0]) * h ** 2) / k

        for n in range(1, nc):
            x_n = x_res[n]
            a_n = 2 + self.p(x_n) * h ** 2
            b_n = self.q(x_n) * h ** 2
            lambdas[n + 1] = -1 / (lambdas[n] - a_n)
            mus[n + 1] = (b_n - mus[n]) / (lambdas[n] - a_n)

        y_res = np.zeros(x_res.shape, dtype=np.float64)

        lambda_n = h ** 2 * self.p(x_res[nc]) / 2 - h * self.beta0 + 1
        mu_n = h ** 2 * self.q(x_res[nc]) / 2 - h * self.beta1

        y_res[nc] = -(mus[nc] - mu_n) / (lambdas[nc] - lambda_n)

        for n in range(nc, 0, -1):
            y_res[n - 1] = lambdas[n] * y_res[n] + mus[n]
        return x_res, y_res


if __name__ == '__main__':
    f = TaskFunc(NUMBER)
    ms = [
        euler_explicit(f),
        runge_kutte(f),
        euler_recalc(f),
        Approx1(f),
        Approx2(f),
        exact_solution(f)
    ]
    FrontEnd(ms).show()
