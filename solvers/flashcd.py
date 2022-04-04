from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from flashcd.estimators import ElasticNet


class Solver(BaseSolver):
    name = 'flashcd'

    install_cmd = 'conda'
    requirements = ['flashcd']
    support_sparse = True

    def set_objective(self, X, y, l1_ratio, lmbda, fit_intercept=False):
        self.X, self.y, self.l1_ratio, self.lmbda = X, y, l1_ratio, lmbda

        self.clf = ElasticNet(
            alpha=lmbda, l1_ratio=l1_ratio, tol=1e-12, fit_intercept=False)
        # Make sure we cache the numba compilation.
        self.run(1)

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.flatten()
