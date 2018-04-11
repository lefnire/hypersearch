import pdb, os, pickle
from sklearn.feature_extraction import DictVectorizer
from box import Box
import pandas as pd
import numpy as np
import xgboost as xgb

def round_(x): return int(round(x))
def bool_(x): return bool(round(x))
def ten_neg(x): return 10**-x
def arr_idx(arr):
    def arr_idx_(x): return arr[int(round(x))]
    return arr_idx_


class Hypersearch(object):
    def __init__(self, data):
        self.data = data

    def search(self):
        klass = self.__class__.__name__
        runs_pkl = f'tmp/runs-{klass}.pkl'
        df = pd.DataFrame()
        vectorizer = DictVectorizer(sparse=False)
        for k, v in self.hypers.items():
            # populate the possible values (only really matters for `str` type). If not present (bool),
            # use [0, 1]
            col = pd.DataFrame({k: v.get('vals', [0, 1])})
            df = pd.concat([df, col], axis=1)  # , ignore_index=True)
        vectorizer.fit(df.ffill().T.to_dict().values())
        feat_names = vectorizer.get_feature_names()
        inverse = lambda vec: vectorizer.inverse_transform([vec])[0]

        bounds = []
        for name in feat_names:
            # Only `str` type features have their names split by DictVectorizer into x=a,x=b,.. In that case we'll
            # make it [0,1] for each option; else we'll use the bounds or true/false
            if name not in self.hypers or self.hypers[name]['type'] == bool_:
                bound = [0, 1]
            else:
                bound = self.hypers[name]['vals']  # Bounded (int,float) b/w min/max
            bounds.append(bound)
        bounds = np.array(bounds)

        # Currently testing XGBoost hyper-search. See ee0af2d for Bayesian Optimization approach
        runs_x, runs_y = [], []
        if os.path.exists(runs_pkl):
            with open(runs_pkl, 'rb') as f:
                runs_x, runs_y = pickle.load(f)

        def rand_hypers(n):
            return np.random.uniform(bounds[:, 0], bounds[:, 1], (n, bounds.shape[0]))
        # First generate random combos so XGB has something basic to work with
        n_init = 10
        randos = rand_hypers(n_init)
        first_guess = vectorizer.transform({k: v['guess'] for k, v in self.hypers.items()})
        randos[0] = first_guess[0]

        for i in range(10000):
            if len(runs_x) < n_init:
                vec = randos[i]
            else:
                if i % 10 == 1:
                    # Still select a rando every so often
                    vec = rand_hypers(1)[0]
                else:
                    n_randos = int(i+1 * 1e3)
                    if i % 10 == 0: n_randos = int(1e7)  # every so often make a really strong guess
                    randos = rand_hypers(n_randos)
                    model = xgb.XGBRegressor()
                    model.fit(runs_x, runs_y)
                    vec = randos[model.predict(randos).argmax()]
                    if i % 10 == 0:
                        # Save winning guess every so often
                        with open(f'tmp/best-{klass}.pkl', 'wb') as f:
                            pickle.dump(inverse(vec), f)
            hypers= inverse(vec)
            score = self.train_predict(**hypers)
            if np.isnan(score): continue
            runs_x.append(vec)
            runs_y.append(score)
            pickle.dump((runs_x, runs_y), open(runs_pkl, 'wb'))
            hypers_print = {k: round(v, 2) for k, v in hypers.items()}
            print(f'{i}. {klass} {hypers_print} score: {round(score, 4)}')

        best_i = np.argmax(runs_y)
        best = self.process_hypers(inverse(runs_x[best_i]))
        return best, runs_y[best_i]

    def process_hypers(self, dirty):
        clean = {}
        str_hypers = {}
        for k, v in dirty.items():
            # For direct matches (float, bool, int) just cast it and add to hypers
            if k in self.hypers:
                cast = self.hypers[k].get('type', bool_)
                clean[k] = cast(v)
            # String-types are more complex. Put them away and deal with next
            else:
                k, option = k.split('=')
                if k not in str_hypers:
                    str_hypers[k] = []
                str_hypers[k].append((option, v))
        # For string hypers, stack them like [[labels], [scores]] and pick the label with the highest score
        for k, arr in str_hypers.items():
            arr = np.array(arr)
            clean[k] = arr[:,0][arr[:,1].argmax()]
            if clean[k] == 'None': clean[k] = None
        for k, v in clean.items():
            if 'process' in self.hypers[k]:
                clean[k] = self.hypers[k]['process'](v)

        # FIXME: not sure how to ensure DictVectorizer inverse False values
        for k, v in self.hypers.items():
            if k not in clean:
                clean[k] = False

        return clean