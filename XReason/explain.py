#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## explain.py
##
##  Created on: Dec 14, 2018
##      Author: Alexey Ignatiev
##      E-mail: alexey.ignatiev@monash.edu
##

#
#==============================================================================
from __future__ import print_function
import numpy as np
import os
from pysat.examples.hitman import Hitman
from pysat.formula import IDPool
from pysmt.shortcuts import Solver
from pysmt.shortcuts import And, BOOL, Implies, Not, Or, Symbol
from pysmt.shortcuts import Equals, GT, Int, Real, REAL, GE, LE
import resource
from six.moves import range
import sys


#
#==============================================================================
class SMTExplainer(object):
    """
        An SMT-inspired minimal explanation extractor for XGBoost models.
    """

    def __init__(self, formula, intvs, imaps, ivars, feats, nof_classes,
            options, xgb):
        """
            Constructor.
        """

        self.feats = feats
        self.intvs = intvs
        self.imaps = imaps
        self.ivars = ivars
        self.nofcl = nof_classes
        self.optns = options
        self.idmgr = IDPool()

        # saving XGBooster
        self.xgb = xgb

        self.verbose = self.optns.verb
        self.oracle = Solver(name=options.solver)
        self.sample_counter = 0

        self.inps = []  # input (feature value) variables
        for f in self.xgb.extended_feature_names_as_array_strings:
            if '_' not in f:
                self.inps.append(Symbol(f, typename=REAL))
            else:
                self.inps.append(Symbol(f, typename=BOOL))

        self.outs = []  # output (class  score) variables
        for c in range(self.nofcl):
            self.outs.append(Symbol('class{0}_score'.format(c), typename=REAL))

        # theory
        self.oracle.add_assertion(formula)

        # current selector
        self.selv = None

    def _add_feature_bounds_constraints(self):
        """
            Bound numeric feature variables so all SAT witnesses stay in-range.
        """
        bounds = getattr(self.xgb, "feature_bounds", None)
        if not bounds:
            return

        for inp in self.inps:
            name = inp.symbol_name()
            if '_' in name:
                continue

            if not name.startswith('f') or not name[1:].isdigit():
                continue

            fid = int(name[1:])
            if fid not in bounds:
                continue

            lo, hi = bounds[fid]
            self.oracle.add_assertion(
                Implies(
                    self.selv,
                    And(
                        GE(inp, Real(float(lo))),
                        LE(inp, Real(float(hi)))
                    )
                )
            )

    def prepare(self, sample):
        """
            Prepare the oracle for computing an explanation.
        """

        if self.selv:
            # disable the previous assumption if any
            self.oracle.add_assertion(Not(self.selv))

        # creating a fresh selector for each call (including repeated samples)
        self.sample_counter += 1
        self.selv = Symbol('sample{0}_selv'.format(self.sample_counter), typename=BOOL)

        self.rhypos = []  # relaxed hypotheses

        # transformed sample
        self.sample = list(self.xgb.transform(sample)[0])

        self.sel2fid = {}  # selectors to original feature ids
        self.sel2vid = {}  # selectors to categorical feature ids

        # preparing the selectors
        for i, (inp, val) in enumerate(zip(self.inps, self.sample), 1):
            feat = inp.symbol_name().split('_')[0]
            selv = Symbol('selv_{0}'.format(feat))
            val = float(val)

            self.rhypos.append(selv)
            if selv not in self.sel2fid:
                self.sel2fid[selv] = int(feat[1:])
                self.sel2vid[selv] = [i - 1]
            else:
                self.sel2vid[selv].append(i - 1)

        # adding relaxed hypotheses to the oracle
        if not self.intvs:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                if '_' not in inp.symbol_name():
                    hypo = Implies(self.selv, Implies(sel, Equals(inp, Real(float(val)))))
                else:
                    hypo = Implies(self.selv, Implies(sel, inp if val else Not(inp)))

                self.oracle.add_assertion(hypo)
        else:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                inp = inp.symbol_name()
                # determining the right interval and the corresponding variable
                for ub, fvar in zip(self.intvs[inp], self.ivars[inp]):
                    if ub == '+' or val < ub:
                        hypo = Implies(self.selv, Implies(sel, fvar))
                        break

                self.oracle.add_assertion(hypo)

        # in case of categorical data, there are selector duplicates
        # and we need to remove them
        self.rhypos = sorted(set(self.rhypos), key=lambda x: int(x.symbol_name()[6:]))

        self._add_feature_bounds_constraints()

        # propagating the true observation
        if self.oracle.solve([self.selv] + self.rhypos):
            model = self.oracle.get_model()
        else:
            assert 0, 'Formula is unsatisfiable under given assumptions'

        # choosing the maximum
        outvals = [float(model.get_py_value(o)) for o in self.outs]
        maxoval = max(zip(outvals, range(len(outvals))))

        # correct class id (corresponds to the maximum computed)
        self.out_id = maxoval[1]
        self.output = self.xgb.target_name[self.out_id]

        # forcing a misclassification, i.e. a wrong observation
        disj = []
        for i in range(len(self.outs)):
            if i != self.out_id:
                disj.append(GT(self.outs[i], self.outs[self.out_id]))
        self.oracle.add_assertion(Implies(self.selv, Or(disj)))

        if self.verbose:
            inpvals = self.xgb.readable_sample(sample)

            self.preamble = []
            for f, v in zip(self.xgb.feature_names, inpvals):
                if f not in v:
                    self.preamble.append('{0} = {1}'.format(f, v))
                else:
                    self.preamble.append(v)

            print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))

    def explain(self, sample, smallest, expl_ext=None, prefer_ext=False):
        """
            Hypotheses minimization.
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # adapt the solver to deal with the current sample
        self.prepare(sample)

        # saving external explanation to be minimized further
        if expl_ext == None or prefer_ext:
            self.to_consider = [True for h in self.rhypos]
        else:
            eexpl = set(expl_ext)
            self.to_consider = [True if i in eexpl else False for i, h in enumerate(self.rhypos)]

        # if satisfiable, then the observation is not implied by the hypotheses
        if self.oracle.solve([self.selv] + [h for h, c in zip(self.rhypos, self.to_consider) if c]):
            print('  no implication!')
            print(self.oracle.get_model())
            sys.exit(1)

        if not smallest:
            self.compute_minimal(prefer_ext=prefer_ext)
        else:
            self.compute_smallest()

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        expl = sorted([self.sel2fid[h] for h in self.rhypos])

        # # if self.verbose:
        # self.preamble = [self.preamble[i] for i in expl]
        # print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.xgb.target_name[self.out_id]))
        # print('  # hypos left:', len(self.rhypos))
        # print('  time: {0:.2f}'.format(self.time))

        return expl

    def explain_contrastive(self, sample, expl_ext=None, prefer_ext=False, include_witness=False):
        """
            Compute a subset-minimal contrastive explanation (CXP), i.e., a
            minimal set of features that need to be allowed to change so that
            a competing class can win.
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # adapt the solver to deal with the current sample
        self.prepare(sample)

        all_rhypos = list(self.rhypos)

        # all fixed but still SAT => empty CXP
        if self.oracle.solve([self.selv] + all_rhypos):
            self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time
            if include_witness:
                return {"features": [], "values": {}}
            return []

        # Maximize fixed hypotheses while preserving SAT. The complement is a CXP.
        fixed = []

        if expl_ext is None:
            candidates = all_rhypos
        else:
            ext = set(expl_ext)
            in_ext = [h for h in all_rhypos if self.sel2fid[h] in ext]
            out_ext = [h for h in all_rhypos if self.sel2fid[h] not in ext]
            candidates = in_ext + out_ext if prefer_ext else in_ext

        for h in candidates:
            trial_fixed = fixed + [h]
            if self.oracle.solve([self.selv] + trial_fixed):
                fixed = trial_fixed

        # If the user restricted features too much, there may be no CXP under restriction.
        if not self.oracle.solve([self.selv] + fixed):
            raise ValueError("No contrastive explanation under the provided expl_ext restriction")

        cxp_hypos = [h for h in all_rhypos if h not in set(fixed)]
        cxp = sorted(set(self.sel2fid[h] for h in cxp_hypos))

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        if not include_witness:
            return cxp

        # model for witness values under fixed hypotheses
        assert self.oracle.solve([self.selv] + fixed)
        model = self.oracle.get_model()
        witness = {}

        for h in cxp_hypos:
            fid = self.sel2fid[h]
            if fid in witness:
                continue

            vids = self.sel2vid[h]
            if len(vids) == 1 and '_' not in self.inps[vids[0]].symbol_name():
                witness[fid] = float(model.get_py_value(self.inps[vids[0]]))
            else:
                chosen_idx = None
                for vid in vids:
                    if int(model.get_py_value(self.inps[vid])) == 1:
                        chosen_idx = vids.index(vid)
                        break

                if chosen_idx is None:
                    continue

                cat_values = self.xgb.categorical_names.get(fid, None)
                if cat_values is not None and chosen_idx < len(cat_values):
                    witness[fid] = cat_values[chosen_idx]
                else:
                    witness[fid] = chosen_idx

        return {"features": cxp, "values": witness}

    def enumerate_axps(self, sample, max_axps=None):
        """
            Enumerate subset-minimal abductive explanations (AXPs).
            SAT means the currently fixed features are insufficient.
            UNSAT means they are sufficient and can be shrunk into an AXP.
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # adapt the solver to deal with the current sample
        self.prepare(sample)

        all_rhypos = list(self.rhypos)
        n = len(all_rhypos)
        universe = [i for i in range(n)]
        axps = []

        with Hitman(bootstrap_with=[universe], htype='lbx') as hitman:
            while True:
                cand = hitman.get()
                if cand is None:
                    break

                fixed_ids = set(cand)
                fixed = [all_rhypos[i] for i in sorted(fixed_ids)]
                is_insufficient = self.oracle.solve([self.selv] + fixed)  # SAT

                if not is_insufficient:
                    # sufficient -> shrink to subset-minimal AXP
                    axp = set(fixed_ids)
                    for i in list(axp):
                        trial = sorted(axp - {i})
                        trial_hypos = [all_rhypos[j] for j in trial]
                        trial_is_insufficient = self.oracle.solve([self.selv] + trial_hypos)
                        if not trial_is_insufficient:  # still sufficient
                            axp.remove(i)

                    axp_list = sorted(axp)
                    axp_feats = sorted(set(self.sel2fid[all_rhypos[i]] for i in axp_list))
                    axps.append(axp_feats)
                    hitman.block(axp_list)

                    if max_axps is not None and len(axps) >= max_axps:
                        break
                else:
                    # insufficient -> compute a minimal CXP and hit it
                    cxp = set(universe) - fixed_ids
                    for i in list(cxp):
                        trial_cxp = cxp - {i}
                        trial_fixed_ids = set(universe) - trial_cxp
                        trial_fixed = [all_rhypos[j] for j in sorted(trial_fixed_ids)]
                        trial_is_insufficient = self.oracle.solve([self.selv] + trial_fixed)
                        if trial_is_insufficient:
                            cxp.remove(i)

                    hitman.hit(sorted(cxp))

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        return axps

    def _build_cxp_witness(self, all_rhypos, cxp_ids, fixed_ids):
        fixed = [all_rhypos[i] for i in sorted(fixed_ids)]
        assert self.oracle.solve([self.selv] + fixed)
        model = self.oracle.get_model()

        witness = {}
        for hid in sorted(cxp_ids):
            h = all_rhypos[hid]
            fid = self.sel2fid[h]
            if fid in witness:
                continue

            vids = self.sel2vid[h]
            if len(vids) == 1 and '_' not in self.inps[vids[0]].symbol_name():
                witness[fid] = float(model.get_py_value(self.inps[vids[0]]))
            else:
                chosen_idx = None
                for vid in vids:
                    if int(model.get_py_value(self.inps[vid])) == 1:
                        chosen_idx = vids.index(vid)
                        break

                if chosen_idx is None:
                    continue

                cat_values = self.xgb.categorical_names.get(fid, None)
                if cat_values is not None and chosen_idx < len(cat_values):
                    witness[fid] = cat_values[chosen_idx]
                else:
                    witness[fid] = chosen_idx

        return witness

    def enumerate_cxps(self, sample, max_cxps=None, include_witness=False):
        """
            Enumerate subset-minimal contrastive explanations (CXPs).
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        self.prepare(sample)

        all_rhypos = list(self.rhypos)
        n = len(all_rhypos)
        universe = [i for i in range(n)]
        cxps = []

        with Hitman(bootstrap_with=[universe], htype='lbx') as hitman:
            while True:
                cand = hitman.get()
                if cand is None:
                    break

                fixed_ids = set(cand)
                fixed = [all_rhypos[i] for i in sorted(fixed_ids)]
                is_insufficient = self.oracle.solve([self.selv] + fixed)  # SAT

                if is_insufficient:
                    # insufficient -> shrink complement to subset-minimal CXP
                    cxp = set(universe) - fixed_ids
                    for i in list(cxp):
                        trial_cxp = cxp - {i}
                        trial_fixed_ids = set(universe) - trial_cxp
                        trial_fixed = [all_rhypos[j] for j in sorted(trial_fixed_ids)]
                        trial_is_insufficient = self.oracle.solve([self.selv] + trial_fixed)
                        if trial_is_insufficient:
                            cxp.remove(i)

                    cxp_list = sorted(cxp)
                    cxp_feats = sorted(set(self.sel2fid[all_rhypos[i]] for i in cxp_list))
                    if include_witness:
                        witness = self._build_cxp_witness(all_rhypos, cxp_list, set(universe) - set(cxp_list))
                        cxps.append({"features": cxp_feats, "values": witness})
                    else:
                        cxps.append(cxp_feats)

                    hitman.hit(cxp_list)

                    if max_cxps is not None and len(cxps) >= max_cxps:
                        break
                else:
                    # sufficient -> compute AXP and block it
                    axp = set(fixed_ids)
                    for i in list(axp):
                        trial = sorted(axp - {i})
                        trial_hypos = [all_rhypos[j] for j in trial]
                        trial_is_insufficient = self.oracle.solve([self.selv] + trial_hypos)
                        if not trial_is_insufficient:
                            axp.remove(i)

                    hitman.block(sorted(axp))

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        return cxps

    def compute_minimal(self, prefer_ext=False):
        """
            Compute any subset-minimal explanation.
        """

        i = 0

        if not prefer_ext:
            # here, we want to reduce external explanation

            # filtering out unnecessary features if external explanation is given
            self.rhypos = [h for h, c in zip(self.rhypos, self.to_consider) if c]
        else:
            # here, we want to compute an explanation that is preferred
            # to be similar to the given external one
            # for that, we try to postpone removing features that are
            # in the external explanation provided

            rhypos  = [h for h, c in zip(self.rhypos, self.to_consider) if not c]
            rhypos += [h for h, c in zip(self.rhypos, self.to_consider) if c]
            self.rhypos = rhypos

        # simple deletion-based linear search
        while i < len(self.rhypos):
            to_test = self.rhypos[:i] + self.rhypos[(i + 1):]

            if self.oracle.solve([self.selv] + to_test):
                i += 1
            else:
                self.rhypos = to_test

    def compute_smallest(self):
        """
            Compute a cardinality-minimal explanation.
        """

        # result
        rhypos = []

        with Hitman(bootstrap_with=[[i for i in range(len(self.rhypos)) if self.to_consider[i]]]) as hitman:
            # computing unit-size MCSes
            for i, hypo in enumerate(self.rhypos):
                if self.to_consider[i] == False:
                    continue

                if self.oracle.solve([self.selv] + self.rhypos[:i] + self.rhypos[(i + 1):]):
                    hitman.hit([i])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 1:
                    print('iter:', iters)
                    print('cand:', hset)

                if self.oracle.solve([self.selv] + [self.rhypos[i] for i in hset]):
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(range(len(self.rhypos))).difference(set(hset)))

                    model = self.oracle.get_model()
                    for h in removed:
                        i = self.sel2fid[self.rhypos[h]]
                        if '_' not in self.inps[i].symbol_name():
                            # feature variable and its expected value
                            var, exp = self.inps[i], self.sample[i]

                            # true value
                            true_val = float(model.get_py_value(var))

                            if not exp - 0.001 <= true_val <= exp + 0.001:
                                unsatisfied.append(h)
                            else:
                                hset.append(h)
                        else:
                            for vid in self.sel2vid[self.rhypos[h]]:
                                var, exp = self.inps[vid], int(self.sample[vid])

                                # true value
                                true_val = int(model.get_py_value(var))

                                if exp != true_val:
                                    unsatisfied.append(h)
                                    break
                            else:
                                hset.append(h)

                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        if self.oracle.solve([self.selv] + [self.rhypos[i] for i in hset] + [self.rhypos[h]]):
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    if self.verbose > 1:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)
                else:
                    self.rhypos = [self.rhypos[i] for i in hset]
                    break