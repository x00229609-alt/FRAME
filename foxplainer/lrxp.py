import pandas as pd
import pickle
import time
from threading import Timer

from .pysat.solvers import Solver


class LRExplainer(object):
    def __init__(self, data, options, feature_bounds=None):
        """
        Parameters
        ----------
        data : ModelDataset
        options : Options
        feature_bounds : dict or list of (min, max) pairs, optional
            Feature bounds used for contrastive explanations.
            Dict keyed by feature index ``{i: (min_val, max_val)}``, or a list
            of ``(min_val, max_val)`` pairs in feature order.
            If omitted, defaults to ``[-1e6, 1e6]`` per feature.
        """
        with open(options.classifier, 'rb') as f:
            self.model = pickle.load(f)
        self.options = options
        self.fnames = data.feature_names
        self.label = data.names[-1]
        self.data = data
        self._feature_bounds = feature_bounds
        self.extract_bounds()

    def extract_bound(self, i):
        values = list(map(lambda l: l[i], self.data.X))
        return max(values), min(values)

    def extract_bounds(self):
        self.lbounds = []
        self.ubounds = []
        coefs = self.model.coef_[0]
        n = len(self.data.extended_feature_names_as_array_strings)
        for i in range(n):
            coef = coefs[i]
            if self._feature_bounds is not None:
                # User-supplied bounds
                if isinstance(self._feature_bounds, dict):
                    lo, hi = self._feature_bounds[i]
                else:
                    lo, hi = self._feature_bounds[i]
                max_value, min_value = hi, lo
            elif self.data.X is not None:
                max_value, min_value = self.extract_bound(i)
            else:
                # No training data and no explicit bounds — use wide defaults
                max_value, min_value = 1e6, -1e6
            if coef >= 0:
                self.lbounds.append(min_value)
                self.ubounds.append(max_value)
            else:
                self.lbounds.append(max_value)
                self.ubounds.append(min_value)
        self.lbounds = pd.to_numeric(pd.Series(self.lbounds, index=self.fnames), errors="raise")
        self.ubounds = pd.to_numeric(pd.Series(self.ubounds, index=self.fnames), errors="raise")

    def _series_value_at(self, series, i):
        if hasattr(series, "iloc"):
            return series.iloc[i]
        return series[i]

    def _set_series_value(self, series, i, value):
        if hasattr(series, "iloc"):
            series.iloc[i] = value
        else:
            series[i] = value

    def _coerce_for_series(self, series, value):
        if hasattr(series, "dtype") and series.dtype != object:
            try:
                return series.dtype.type(value)
            except (TypeError, ValueError):
                pass
        return value

    def free_attr(self, i, inst, lbounds, ubounds, deset, inset):
        self.inst = inst
        self._set_series_value(lbounds, i, self._coerce_for_series(lbounds, self._series_value_at(self.lbounds, i)))
        self._set_series_value(ubounds, i, self._coerce_for_series(ubounds, self._series_value_at(self.ubounds, i)))
        deset.remove(i)
        inset.add(i)

    def fix_attr(self, i, inst, lbounds, ubounds, deset, inset):
        value = self._series_value_at(inst, i)
        self._set_series_value(lbounds, i, self._coerce_for_series(lbounds, value))
        self._set_series_value(ubounds, i, self._coerce_for_series(ubounds, value))
        deset.remove(i)
        inset.add(i)

    def equal_pred(self, lbounds, ubounds):
        return self._predict_row(lbounds) == self._predict_row(ubounds)

    def _bound_value(self, bounds, i):
        return self._series_value_at(bounds, i)

    def _row_value(self, row, i):
        return self._series_value_at(row, i)

    def _row_to_frame(self, row):
        if isinstance(row, pd.Series):
            return row.to_frame().T[self.fnames]
        return pd.DataFrame([row], columns=self.fnames)

    def _predict_row(self, row):
        return self.model.predict(self._row_to_frame(row))[0]

    def _other_class(self, pred):
        classes = list(getattr(self.model, "classes_", []))
        if len(classes) == 2:
            return classes[0] if pred == classes[1] else classes[1]
        for cls in classes:
            if cls != pred:
                return cls
        return pred

    def _find_cxp_witness(self, inst, exp, pred):
        """
            Find concrete witness values for a CXP by moving the selected
            features to the score-extremizing bounds for the opposite class.
        """
        exp_sorted = sorted(exp)
        if not exp_sorted:
            return {}

        cand = inst.copy()
        use_lower_bounds = pred == self.model.classes_[1]
        for i in exp_sorted:
            bound = self._bound_value(self.lbounds if use_lower_bounds else self.ubounds, i)
            self._set_series_value(cand, i, self._coerce_for_series(cand, bound))

        cand_pred = self._predict_row(cand)
        if cand_pred != pred:
            return {i: self._row_value(cand, i) for i in exp_sorted}, cand_pred

        # Fallback: return endpoint values and best-effort opposite class.
        return {i: self._bound_value(self.lbounds, i) for i in exp_sorted}, self._other_class(pred)

    def explain(self, inst):
        self.hypos = list(range(len(inst)))
        pred = self._predict_row(inst)
        self.time = {'abd': 0, 'con': 0}
        self.exps = {'abd': [], 'con': []}
        if self.options.xnum not in (-1, 'all'):
            if self.options.xtype in ['abd', 'abductive']:
                self.exps['abd'].append(self.extract_AXp(inst))
            else:
                self.exps['con'].append(self.extract_CXp(inst))
        else:
            self.exps = self.enumrate(inst)


        preamble = ['{0} = {1}'.format(self.fnames[i], self._row_value(inst, i)) for i in self.hypos]
        explained_instance = 'IF {0} THEN {1} = {2}'.format(' AND '.join(preamble), self.label, pred)

        explanation_list =  {'abd': [], 'con': []}
        explanation_size_list =  {'abd': [], 'con': []}

        #xtype = 'abd' if self.options.xtype in ['abd', 'abductive'] else 'con'
        for xtype in ['abd', 'con']:
            for exp in self.exps[xtype]:
                if xtype == 'con':
                    witness, contrast_pred = self._find_cxp_witness(inst, exp, pred)
                    preamble = ['{0} = {1}'.format(self.fnames[i], witness.get(i, self._row_value(inst, i)))
                                for i in sorted(exp)]
                else:
                    preamble = ['{0} = {1}'.format(self.fnames[i], self._row_value(inst, i))
                                for i in sorted(exp)]
                explanation = 'IF {} THEN {} {} {}'.format(' AND '.join(preamble),
                                                                self.label,
                                                                '=',
                                                                pred if xtype == 'abd' else contrast_pred)
                explanation_size = 'Number of Explained Features: {0}'.format(len(exp))
                explanation_list[xtype].append(explanation)
                explanation_size_list[xtype].append(explanation_size)

                """
                xtype_ = 'abd' if xtype == 'con' else 'con'
                
                for exp_ in self.exps[xtype_]:
                    preamble = ['{0} {1} {2}'.format(self.fnames[i], '=' if xtype_ == 'abd' else '!=', self._row_value(inst, i))
                                for i in sorted(exp_)]
                    print_xtype = 'Abductive Explanation' if xtype_ == 'abd' else 'Contrastive Explanation'
                    print('{}:\nIF {}\nTHEN {} {} {}'.format(print_xtype,
                                                             ' AND \n'.join(preamble),
                                                             self.label,
                                                             '=' if xtype_ == 'abd' else '!=',
                                                             pred))
                    print('Explanation Size: {0}'.format(len(exp_)))
                """
        return self.exps, self.time, explained_instance, explanation_list, explanation_size_list, pred

    def extract_AXp(self, inst, seed=set()):
        lbounds = inst.copy()
        ubounds = inst.copy()
        candidate, drop, pick = set(self.hypos), set(), set()
        for i in seed:
            self.free_attr(i, inst, lbounds, ubounds, candidate, drop)
        potential = list(filter(lambda l: l not in seed, self.hypos))
        for i in potential:
            self.free_attr(i, inst, lbounds, ubounds, candidate, drop)
            if not self.equal_pred(lbounds, ubounds):
                self.fix_attr(i, inst, lbounds, ubounds, drop, pick)
        return pick

    def extract_CXp(self, inst, seed=set()):
        lbounds = self.lbounds.copy()
        ubounds = self.ubounds.copy()
        candidate, drop, pick = set(self.hypos), set(), set()
        for i in seed:
            self.fix_attr(i, inst, lbounds, ubounds, candidate, drop)
        potential = list(filter(lambda l: l not in seed, self.hypos))
        for i in potential:
            self.fix_attr(i, inst, lbounds, ubounds, candidate, drop)
            if self.equal_pred(lbounds, ubounds):
                self.free_attr(i, inst, lbounds, ubounds, drop, pick)
        return pick


    # def enumrate_og(self, inst):
    #     oracle = Solver(name=self.options.solver)
    #     exps = {'abd': [], 'con': []}
    #     self.hit = set()
    #     while True:
    #         if not oracle.solve():
    #             return exps
    #         assignment = oracle.get_model()
    #         lbounds = self.lbounds.copy()
    #         ubounds = self.ubounds.copy()
    #         for i in self.hit:
    #             if assignment[i] > 0:
    #                 lbounds[i] = inst[i]
    #                 ubounds[i] = inst[i]
    #         if self.equal_pred(lbounds, ubounds):
    #             seed = set(self.hypos).difference(set(filter(lambda i: assignment[i] > 0, self.hit)))
    #             exp = self.extract_AXp(inst, seed)
    #             exps['abd'].append(exp)
    #             oracle.add_clause([-(i + 1) for i in sorted(exp)])
    #         else:
    #             seed = set(filter(lambda i: assignment[i] > 0, self.hit))
    #             exp = self.extract_CXp(inst, seed)
    #             exps['con'].append(exp)
    #             oracle.add_clause([i + 1 for i in sorted(exp)])
    #         self.hit.update(exp)

    def enumrate(self, inst):
        def interrupt(s):
            timed_out[0] = True
            s.interrupt()

        timed_out = [False]
        oracle = Solver(name=self.options.solver)
        exps = {'abd': [], 'con': []}
        self.hit = set()
        if self.options.time_limit is not None:
            timer = Timer(self.options.time_limit, interrupt, [oracle])
            timer.start()
        try:
            while True:
                if timed_out[0]:
                    return exps
                status = oracle.solve_limited(expect_interrupt=True) if self.options.time_limit is not None else oracle.solve()

                if status is False:
                    # UNSAT — natural end of enumeration
                    return exps
                elif status is None:
                    # Interrupted by timer — return whatever we have so far
                    return exps

                assignment = oracle.get_model()
                lbounds = self.lbounds.copy()
                ubounds = self.ubounds.copy()
                for i in self.hit:
                    if assignment[i] > 0:
                        self._set_series_value(lbounds, i, self._coerce_for_series(lbounds, self._series_value_at(inst, i)))
                        self._set_series_value(ubounds, i, self._coerce_for_series(ubounds, self._series_value_at(inst, i)))
                if self.equal_pred(lbounds, ubounds):
                    seed = set(self.hypos).difference(set(filter(lambda i: assignment[i] > 0, self.hit)))
                    exp = self.extract_AXp(inst, seed)
                    exps['abd'].append(exp)
                    oracle.add_clause([-(i + 1) for i in sorted(exp)])
                    # print("new abductive")
                else:
                    seed = set(filter(lambda i: assignment[i] > 0, self.hit))
                    exp = self.extract_CXp(inst, seed)
                    exps['con'].append(exp)
                    oracle.add_clause([i + 1 for i in sorted(exp)])
                    # print("new contrastive")
                self.hit.update(exp)
        finally:
            # Always cancel the timer — whether we finished naturally,
            # were interrupted, or hit an unexpected exception
            if self.options.time_limit is not None:
                if timer is not None:
                    timer.cancel()
