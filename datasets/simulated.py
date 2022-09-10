from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from benchopt.datasets import make_correlated_data


class Dataset(BaseDataset):

    name = "simulated"

    parameters = {
        'n_samples, n_features': [
            (500, 3000),
            (500, 1000),
            (3000, 5000)
        ],
        'X_density': [1., 0.1]
    }

    def __init__(self, n_samples=10, n_features=50, X_density=1., random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.X_density = X_density
        self.random_state = random_state

    def get_data(self):
        X, y, _ = make_correlated_data(self.n_samples, self.n_features,
                                       random_state=self.random_state,
                                       X_density=self.X_density)
        y = 2 * (y > 0) - 1

        data = dict(X=X, y=y)

        return data
