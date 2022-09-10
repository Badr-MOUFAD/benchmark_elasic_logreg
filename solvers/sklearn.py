import warnings
from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression


class Solver(BaseSolver):

    name = 'scikit-learn'

    def set_objective(self, X, y, lmbd, l1_ratio):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.l1_ratio = l1_ratio

        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.clf = LogisticRegression(
            solver='saga', C=1 / self.lmbd,
            penalty='elasticnet', fit_intercept=False,
            l1_ratio=self.l1_ratio, tol=1e-9)

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.flatten()
