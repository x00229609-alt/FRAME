#!/us/bin/env python
#-*- coding:utf-8 -*-
##
## xgbooster.py
##
##  Created on: Dec 7, 2018
##      Author: Nina Narodytska, Alexey Ignatiev
##      E-mail: narodytska@vmware.com, alexey.ignatiev@monash.edu
##

#
#==============================================================================
from __future__ import print_function
from .validate import SMTValidator
from .encode import SMTEncoder
from .explain import SMTExplainer
from pathlib import Path
from itertools import product
from html import escape
import numpy as np
import os
import base64
import resource
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn
# print('The scikit-learn version is {}.'.format(sklearn.__version__))

from  sklearn.preprocessing import OneHotEncoder
import sys
from six.moves import range
from .tree import TreeEnsemble
import xgboost as xgb
from xgboost import XGBClassifier, Booster
import pickle
from numbers import Number


#
#==============================================================================
class XGBooster(object):
    """
        The main class to train/encode/explain XGBoost models.
    """

    def __init__(self, use_categorical=False, from_model=None, categorical_values_dict=None, stakeholder_name='developer',
                 class_value_map=None, feature_bounds=None):
        """
            Constructor.
        """
        self.init_stime = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        self.init_ctime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime

        # non-CLI defaults
        self.use_categorical = bool(use_categorical)
        self.output = '.'
        self.files = [from_model if from_model is not None else 'model']
        self.smallest = False
        self.verb = 0
        self.solver = 'z3'
        self.seed = 14
        self.stakeholder_type = str(stakeholder_name).strip().lower()
        self.class_value_map = class_value_map if class_value_map is not None else {}
        self.feature_bounds = {}
        # self.encode = 'smt'
        np.random.seed(14)

        if from_model is None:
            raise ValueError("from_model path must be provided")

        self.load_datainfo(from_model)
        if categorical_values_dict is not None:
            self._apply_categorical_values_dict(categorical_values_dict)
        if feature_bounds is not None:
            self._apply_feature_bounds_dict(feature_bounds)

        # create extra file names
        try:
            os.stat(self.output)
        except:
            os.mkdir(self.output)

        self.mapping_features()
        #################
        self.test_encoding_transformes()

        # bench_name = os.path.splitext(os.path.basename(self.files[0]))[0]
        # bench_dir_name = self.output + "/" + bench_name
        # try:
        #     os.stat(bench_dir_name)
        # except:
        #     os.mkdir(bench_dir_name)
        #
        # self.basename = (os.path.join(bench_dir_name, bench_name +
        #                 "_nbestim_" + str(self.n_estimators) +
        #                 "_maxdepth_" + str(self.maxdepth) +
        #                 "_testsplit_" + str(self.testsplit)))
        #
        # data_suffix =  '.splitdata.pkl'
        # self.modfile =  self.basename + '.mod.pkl'
        #
        # self.mod_plainfile =  self.basename + '.mod.txt'
        #
        # self.resfile =  self.basename + '.res.txt'

        model_path = Path(from_model)
        self.basename = str(model_path.with_name("model"))
        self.encfile =  self.basename + '.enc.txt'

        self._ensure_smt_encoding()
        # self.expfile =  self.basename + '.exp.txt'

    def _ensure_smt_encoding(self):
        """
        Ensure SMT encoding and its companion metadata are initialized.
        """
        if all(hasattr(self, k) for k in ("enc", "intvs", "imaps", "ivars")):
            return
        encoder = SMTEncoder(self.model, self.feature_names, self.num_class, self)
        self.enc, self.intvs, self.imaps, self.ivars = encoder.encode()

    def _sync_feature_metadata_with_model(self):
        """
        Keep loaded feature metadata consistent with the actual trained model width.
        """
        model_n = getattr(self.model, "n_features_in_", None)
        if model_n is None:
            return
        model_n = int(model_n)
        if self.nb_features == model_n:
            return

        if len(self.feature_names) >= model_n:
            self.feature_names = list(self.feature_names[:model_n])
        else:
            self.feature_names = list(self.feature_names) + [f"f{i}" for i in range(len(self.feature_names), model_n)]
        self.nb_features = model_n

        self.categorical_features = sorted([int(i) for i in self.categorical_features if int(i) < self.nb_features])
        self.categorical_names = {int(i): v for i, v in self.categorical_names.items() if int(i) < self.nb_features}
        if isinstance(self.encoder, dict):
            self.encoder = {int(i): enc for i, enc in self.encoder.items() if int(i) < self.nb_features}
        if isinstance(self.feature_bounds, dict):
            normalized = {}
            for raw_key, raw_bound in self.feature_bounds.items():
                idx = self._resolve_feature_key_to_index(raw_key, "loaded feature_bounds")
                normalized[idx] = self._parse_feature_bound(raw_bound, raw_key)
            self.feature_bounds = normalized

    def _resolve_feature_key_to_index(self, raw_key, source_name):
        if isinstance(raw_key, int):
            idx = raw_key
        elif isinstance(raw_key, str) and raw_key.startswith('f') and raw_key[1:].isdigit():
            idx = int(raw_key[1:])
        elif isinstance(raw_key, str):
            key = raw_key.strip()
            name_to_idx = {}
            for i, name in enumerate(self.feature_names):
                s = str(name)
                name_to_idx[s] = i
                name_to_idx[s.lower()] = i
            if key in name_to_idx:
                idx = name_to_idx[key]
            elif key.lower() in name_to_idx:
                idx = name_to_idx[key.lower()]
            else:
                raise ValueError("Unknown feature key in {0}: {1}".format(source_name, raw_key))
        else:
            raise ValueError("Unknown feature key in {0}: {1}".format(source_name, raw_key))

        if idx < 0 or idx >= self.nb_features:
            raise ValueError("Feature index out of range in {0}: {1}".format(source_name, idx))
        return int(idx)

    def _parse_feature_bound(self, raw_bound, raw_key):
        if isinstance(raw_bound, dict):
            if "min" not in raw_bound or "max" not in raw_bound:
                raise ValueError("Feature bound for {0} must contain 'min' and 'max'".format(raw_key))
            lo, hi = raw_bound["min"], raw_bound["max"]
        elif isinstance(raw_bound, (list, tuple, np.ndarray)) and len(raw_bound) == 2:
            lo, hi = raw_bound[0], raw_bound[1]
        else:
            raise ValueError(
                "Feature bound for {0} must be (min, max) or {{'min': v1, 'max': v2}}".format(raw_key)
            )

        if not isinstance(lo, Number) or not isinstance(hi, Number):
            raise ValueError("Feature bounds for {0} must be numeric".format(raw_key))
        lo, hi = float(lo), float(hi)
        if lo > hi:
            raise ValueError("Feature bound for {0} has min > max".format(raw_key))
        return lo, hi

    def _apply_feature_bounds_dict(self, feature_bounds):
        """
        Accept optional numeric feature bounds:
            {feature_idx|feature_name|fN: (min, max)} or {'min': m, 'max': M}
        """
        if not isinstance(feature_bounds, dict):
            raise ValueError("feature_bounds must be a dict")

        parsed = {}
        for raw_key, raw_bound in feature_bounds.items():
            idx = self._resolve_feature_key_to_index(raw_key, "feature_bounds")
            parsed[idx] = self._parse_feature_bound(raw_bound, raw_key)
        self.feature_bounds = parsed

    def _validate_feature_bounds(self, sample_arr):
        if not self.feature_bounds:
            return

        x = np.array(sample_arr, copy=False)
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        for row_idx, row in enumerate(x):
            for fid, (lo, hi) in self.feature_bounds.items():
                if fid in self.categorical_features:
                    continue
                value = row[int(fid)]
                if isinstance(value, (np.bool_, bool)):
                    value = int(value)
                if not isinstance(value, Number):
                    raise ValueError(
                        "Feature '{0}' must be numeric to check bounds; got value '{1}'".format(
                            self.feature_names[int(fid)], value
                        )
                    )
                fv = float(value)
                if fv < lo or fv > hi:
                    raise ValueError(
                        "Feature '{0}' out of bounds in row {1}: got {2}, expected within [{3}, {4}]".format(
                            self.feature_names[int(fid)], row_idx, fv, lo, hi
                        )
                    )

    def _coerce_sample_array(self, sample):
        """
        Convert user sample into model-aligned numpy array while preserving column order.
        """
        s = sample
        if hasattr(s, "columns"):  # pandas DataFrame
            if self.feature_names:
                missing = [f for f in self.feature_names if f not in s.columns]
                if missing:
                    raise ValueError("Sample is missing required features: {0}".format(missing[:5]))
                s = s.loc[:, self.feature_names]
            s = s.to_numpy()
        elif hasattr(s, "index") and hasattr(s, "reindex") and hasattr(s, "to_numpy"):  # pandas Series
            if self.feature_names:
                s = s.reindex(self.feature_names)
            s = s.to_numpy()
        else:
            s = np.array(s)
        return np.array(s)

    def _predict_single_row(self, row):
        """
        Predict one sample row, preserving categorical dtypes when needed.
        """
        row = np.array(row, dtype=object)
        if row.ndim == 0:
            raise ValueError("row must contain feature values, got scalar")
        if row.ndim > 1:
            row = row.reshape(-1)
        if row.shape[0] != len(self.feature_names):
            raise ValueError(
                "row has {0} features but model expects {1}".format(row.shape[0], len(self.feature_names))
            )

        if self.use_categorical and len(self.categorical_features) > 0:
            try:
                import pandas as pd
            except ImportError as exc:
                raise RuntimeError("pandas is required for categorical XGBoost prediction") from exc

            # Mirror direct model inference path: align to booster feature order and
            # infer categories from values instead of forcing category domains.
            df = pd.DataFrame([row], columns=self.feature_names)
            booster_names = None
            if hasattr(self.model, "get_booster"):
                booster = self.model.get_booster()
                booster_names = booster.feature_names
            if booster_names:
                missing = [c for c in booster_names if c not in df.columns]
                if missing:
                    raise ValueError("Sample is missing required model features: {0}".format(missing[:5]))
                df = df.loc[:, booster_names]
            for c in df.columns:
                if df[c].dtype.name == "object":
                    df[c] = df[c].astype("category")
            return self.model.predict(df)[0]
        return self.model.predict(np.expand_dims(row, axis=0))[0]

    def _apply_categorical_values_dict(self, categorical_values_dict):
        """
        Accept VeriX-style spec:
            {feature_idx|feature_name|fN: [values...]}
        """
        if not isinstance(categorical_values_dict, dict):
            raise ValueError("categorical_values_dict must be a dict")

        # Build robust name->index aliases (exact + case-insensitive + model names)
        name_to_idx = {}
        for i, name in enumerate(self.feature_names):
            s = str(name)
            name_to_idx[s] = i
            name_to_idx[s.lower()] = i

        booster_names = None
        if hasattr(self.model, "get_booster"):
            booster = self.model.get_booster()
            booster_names = booster.feature_names
        if booster_names:
            for i, name in enumerate(booster_names):
                s = str(name)
                name_to_idx[s] = i
                name_to_idx[s.lower()] = i
        categorical_names = {}
        categorical_features = []

        for raw_key, raw_values in categorical_values_dict.items():
            if isinstance(raw_key, int):
                idx = raw_key
            elif isinstance(raw_key, str) and raw_key.startswith('f') and raw_key[1:].isdigit():
                idx = int(raw_key[1:])
            elif isinstance(raw_key, str):
                key = raw_key.strip()
                if key in name_to_idx:
                    idx = name_to_idx[key]
                elif key.lower() in name_to_idx:
                    idx = name_to_idx[key.lower()]
                else:
                    raise ValueError("Unknown categorical feature key: {0}".format(raw_key))
            else:
                raise ValueError("Unknown categorical feature key: {0}".format(raw_key))

            if idx < 0 or idx >= self.nb_features:
                print(self.nb_features, idx)
                raise ValueError("Categorical feature index out of range: {0}".format(idx))

            values = list(raw_values) if isinstance(raw_values, (list, tuple, np.ndarray)) else [raw_values]
            if not values:
                raise ValueError("Empty categorical values for feature: {0}".format(raw_key))

            categorical_features.append(idx)
            categorical_names[idx] = values

        self.use_categorical = len(categorical_features) > 0
        self.categorical_features = sorted(categorical_features)
        self.categorical_names = {i: categorical_names[i] for i in self.categorical_features}
        self.encoder = {}
        for i in self.categorical_features:
            vals = np.asarray(self.categorical_names[i]).reshape(-1, 1)
            categories = [np.asarray(self.categorical_names[i])]
            try:
                # sklearn >= 1.2
                enc = OneHotEncoder(categories=categories, sparse_output=False, handle_unknown='ignore')
            except TypeError:
                # sklearn < 1.2
                enc = OneHotEncoder(categories=categories, sparse=False, handle_unknown='ignore')
            enc.fit(vals)
            self.encoder[i] = enc

    def form_datefile_name(self, modfile):
        data_suffix =  '.splitdata.pkl'
        return  modfile + data_suffix

    def pickle_save_file(self, filename, data):
        try:
            with open(filename, "wb") as f:
                pickle.dump(data, f)
        except OSError as exc:
            raise RuntimeError("Cannot save to file {0}: {1}".format(filename, exc)) from exc

    def pickle_load_file(self, filename):
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError as exc:
            raise RuntimeError("Cannot load from file {0}: file not found".format(filename)) from exc
        except EOFError as exc:
            fsize = os.path.getsize(filename) if os.path.exists(filename) else 0
            raise RuntimeError(
                "Cannot load from file {0}: unexpected EOF (size={1} bytes)".format(filename, fsize)
            ) from exc
        except (pickle.UnpicklingError, OSError) as exc:
            raise RuntimeError("Cannot load from file {0}: {1}".format(filename, exc)) from exc

    def save_datainfo(self, filename):

        print("saving  model to ", filename)
        self.pickle_save_file(filename, self.model)

        filename_data = self.form_datefile_name(filename)
        print("saving  data to ", filename_data)
        samples = {}
        samples["X"] = self.X
        samples["Y"] = self.Y
        samples["X_train"] = self.X_train
        samples["Y_train"] = self.Y_train
        samples["X_test"] = self.X_test
        samples["Y_test"] = self.Y_test
        samples["feature_names"] = self.feature_names
        samples["target_name"] = self.target_name
        samples["num_class"] = self.num_class
        samples["categorical_features"] = self.categorical_features
        samples["categorical_names"] = self.categorical_names
        samples["encoder"] = self.encoder
        samples["use_categorical"] = self.use_categorical
        samples["feature_bounds"] = self.feature_bounds


        self.pickle_save_file(filename_data, samples)

    def load_datainfo(self, filename):
        print("loading model from ", filename)
        self.model = XGBClassifier()
        self.model = self.pickle_load_file(filename)

        datafile = self.form_datefile_name(filename)
        print("loading datainfo from ", datafile)
        if not os.path.exists(datafile):
            print("splitdata file not found, generating:", datafile)
            self._generate_splitdata_from_model(datafile)

        print("loading data from ", datafile)
        loaded_data = self.pickle_load_file(datafile)
        self.X = loaded_data.get("X", np.empty((0, 0), dtype=np.float32))
        self.Y = loaded_data.get("Y", np.empty((0,), dtype=np.float32))
        self.X_train = loaded_data.get("X_train", np.empty((0, 0), dtype=np.float32))
        self.X_test = loaded_data.get("X_test", np.empty((0, 0), dtype=np.float32))
        self.Y_train = loaded_data.get("Y_train", np.empty((0,), dtype=np.float32))
        self.Y_test = loaded_data.get("Y_test", np.empty((0,), dtype=np.float32))
        self.feature_names = loaded_data.get("feature_names", [])
        self.target_name = loaded_data.get("target_name", [])
        self.num_class = loaded_data.get("num_class", len(self.target_name) if self.target_name else 2)
        self.nb_features = len(self.feature_names)
        self.categorical_features = loaded_data.get("categorical_features", [])
        self.categorical_names = loaded_data.get("categorical_names", {})
        self.encoder = loaded_data.get("encoder", {})
        self.use_categorical = loaded_data.get("use_categorical", False)
        loaded_feature_bounds = loaded_data.get("feature_bounds", {}) or {}
        if isinstance(loaded_feature_bounds, dict):
            self.feature_bounds = loaded_feature_bounds
        else:
            self.feature_bounds = {}
        self._sync_feature_metadata_with_model()

    def _generate_splitdata_from_model(self, datafile):
        """
        Generate a minimal .splitdata.pkl when only a model file is provided.
        """
        n_features = getattr(self.model, "n_features_in_", None)
        booster_names = None
        if hasattr(self.model, "get_booster"):
            booster = self.model.get_booster()
            booster_names = booster.feature_names
        if n_features is None:
            if booster_names is not None:
                n_features = len(booster_names)
            else:
                n_features = 0

        if booster_names is not None and len(booster_names) == int(n_features):
            feature_names = [str(f) for f in booster_names]
        else:
            feature_names = [f"f{i}" for i in range(int(n_features))]
        classes = getattr(self.model, "classes_", None)
        if classes is not None and len(classes) > 0:
            target_name = [str(c) for c in classes]
            num_class = len(target_name)
        else:
            target_name = ["0", "1"]
            num_class = 2

        empty_X = np.empty((0, int(n_features)), dtype=np.float32)
        empty_Y = np.empty((0,), dtype=np.float32)
        samples = {
            "X": empty_X,
            "Y": empty_Y,
            "X_train": empty_X,
            "Y_train": empty_Y,
            "X_test": empty_X,
            "Y_test": empty_Y,
            "feature_names": feature_names,
            "target_name": target_name,
            "num_class": num_class,
            "categorical_features": [],
            "categorical_names": {},
            "encoder": {},
            "use_categorical": False,
            "feature_bounds": {},
        }
        self.pickle_save_file(datafile, samples)

    def train(self, outfile=None):
        """
            Train a tree ensemble using XGBoost.
        """

        return self.build_xgbtree(outfile)

    def encode(self, test_on=None):
        """
            Encode a tree ensemble trained previously.
        """

        encoder = SMTEncoder(self.model, self.feature_names, self.num_class, self)
        self.enc, self.intvs, self.imaps, self.ivars = encoder.encode()

        if test_on:
            encoder.test_sample(np.array(test_on))

        encoder.save_to(self.encfile)

    def explain(self, sample,
            expl_ext=None, prefer_ext=False, nof_feats=5,
            in_jupyter=True,
            prediction_label="Prediction",
            ffa_graph_dir=".",
            class_value_map=None):
        """
            Explain a prediction made for a given sample with a previously
            trained tree ensemble.
            Explanation mode is derived from stakeholder_type.
        """
        # Derive explanation behavior from stakeholder type only.
        if self.stakeholder_type == 'developer':
            enumerate_all = True
            max_items = None
            include_witness = True
            include_ffa_graph = True
        elif self.stakeholder_type == 'decision_maker':
            enumerate_all = True
            max_items = None
            include_witness = True
            include_ffa_graph = False
        elif self.stakeholder_type == 'model_subject':
            enumerate_all = True
            max_items = 2
            include_witness = True
            include_ffa_graph = False
        else:
            enumerate_all = False
            max_items = 1
            include_witness = True
            include_ffa_graph = False
        
        self._ensure_smt_encoding()
        if 'x' not in dir(self):
            self.x = SMTExplainer(self.enc, self.intvs, self.imaps,
                    self.ivars, self.feature_names, self.num_class,
                    self, self)

        def idx_to_name(idx):
            i = int(idx)
            if i >= 0 and i < len(self.feature_names):
                return str(self.feature_names[i])
            return "f{0}".format(i)

        def normalize_expl_ext(expl_ext):
            if expl_ext is None:
                return None
            if not isinstance(expl_ext, (list, tuple, set)):
                return expl_ext

            name_to_idx = {}
            for i, name in enumerate(self.feature_names):
                s = str(name)
                name_to_idx[s] = i
                name_to_idx[s.lower()] = i

            result = []
            for f in expl_ext:
                if isinstance(f, (int, np.integer)):
                    result.append(int(f))
                elif isinstance(f, str):
                    key = f.strip()
                    if key.startswith('f') and key[1:].isdigit():
                        result.append(int(key[1:]))
                    elif key in name_to_idx:
                        result.append(name_to_idx[key])
                    elif key.lower() in name_to_idx:
                        result.append(name_to_idx[key.lower()])
                    else:
                        raise ValueError("Unknown feature in expl_ext: {0}".format(f))
                else:
                    raise ValueError("Unsupported feature reference in expl_ext: {0}".format(f))
            return result

        def map_named_features(result):
            if isinstance(result, list):
                if result and isinstance(result[0], dict):
                    return [map_named_features(r) for r in result]
                if result and isinstance(result[0], (list, tuple, set)):
                    return [[idx_to_name(i) for i in expl] for expl in result]
                return [idx_to_name(i) for i in result]

            if isinstance(result, dict) and "features" in result:
                named = dict(result)
                named["features"] = [idx_to_name(i) for i in result.get("features", [])]
                if "values" in result and isinstance(result["values"], dict):
                    named["values"] = {
                        idx_to_name(k): v for k, v in result["values"].items()
                    }
                return named

            if isinstance(result, dict) and "axps" in result and "ffa" in result:
                return {
                    "axps": map_named_features(result["axps"]),
                    "ffa": {idx_to_name(k): v for k, v in result["ffa"].items()}
                }

            if isinstance(result, dict) and "abd" in result and "con" in result:
                named = {
                    "abd": map_named_features(result["abd"]),
                    "con": map_named_features(result["con"])
                }
                if "ffa" in result and isinstance(result["ffa"], dict):
                    named["ffa"] = {idx_to_name(k): v for k, v in result["ffa"].items()}
                return named

            return result

        def refine_contrastive_witness(result, sample_arr):
            if not isinstance(result, dict) or "features" not in result or "values" not in result:
                return result

            feat_idxs = [int(i) for i in result.get("features", [])]
            if not feat_idxs:
                result["verified"] = False
                return result

            base = sample_arr[0].copy() if sample_arr.ndim > 1 else sample_arr.copy()
            base_pred = self._predict_single_row(base)

            def flips(assignments):
                cand = np.array(base, copy=True)
                for fid, val in assignments.items():
                    cand[int(fid)] = val
                pred = self._predict_single_row(cand)
                return pred != base_pred

            def search_domains(target_feats, max_tries):
                domains = []
                for fid in target_feats:
                    domain = []
                    if fid in self.categorical_features:
                        domain = list(self.categorical_names.get(fid, []))

                    if fid in suggested and suggested[fid] not in domain:
                        domain.insert(0, suggested[fid])

                    if not domain:
                        return None
                    domains.append(domain)

                tries = 0
                for combo in product(*domains):
                    tries += 1
                    if tries > max_tries:
                        break

                    assignment = {fid: val for fid, val in zip(target_feats, combo)}
                    if all(assignment[fid] == base[fid] for fid in target_feats):
                        continue

                    if flips(assignment):
                        return assignment
                return None

            # First try the SMT witness directly.
            suggested = {}
            values = result.get("values", {})
            if isinstance(values, dict):
                for fid in feat_idxs:
                    if fid in values:
                        suggested[fid] = values[fid]
            if len(suggested) == len(feat_idxs) and flips(suggested):
                result["verified"] = True
                return result

            # Search over declared categorical domains for the computed CXP.
            assignment = search_domains(feat_idxs, max_tries=20000)
            if assignment is not None:
                result["values"] = assignment
                result["verified"] = True
                return result

            # Fallback: try adding one extra categorical feature and search again.
            remaining = [f for f in self.categorical_features if f not in feat_idxs]
            for extra in remaining:
                target_feats = feat_idxs + [extra]
                assignment = search_domains(target_feats, max_tries=40000)
                if assignment is not None:
                    result["features"] = sorted(target_feats)
                    result["values"] = assignment
                    result["verified"] = True
                    result["augmented"] = True
                    return result

            result["verified"] = False
            return result

        sample = self._coerce_sample_array(sample)
        self._validate_feature_bounds(sample)
        expl_ext = normalize_expl_ext(expl_ext)

        def compute_ffa_from_axps(axps):
            if not axps:
                return {}

            score = {}
            for axp in axps:
                if not axp:
                    continue
                w = 1.0 / float(len(axp))
                for f in axp:
                    f = int(f)
                    score[f] = score.get(f, 0.0) + w

            nof_axps = float(len(axps))
            score = {f: (v / nof_axps) for f, v in score.items()}
            return dict(sorted(score.items(), key=lambda kv: (-kv[1], int(kv[0]))))

        def _is_verified_cxp(cxp):
            return isinstance(cxp, dict) and cxp.get("verified") is True

        def _cxp_size(cxp):
            if isinstance(cxp, dict):
                return len(cxp.get("features", []))
            try:
                return len(cxp)
            except TypeError:
                return 10 ** 9

        abd_expl = self.x.enumerate_axps(sample, max_axps=max_items) if enumerate_all \
            else self.x.explain(sample, self.smallest, expl_ext, prefer_ext)
        con_expl = self.x.enumerate_cxps(sample, max_cxps=max_items, include_witness=include_witness) if enumerate_all \
            else self.x.explain_contrastive(sample, expl_ext=expl_ext, prefer_ext=prefer_ext, include_witness=include_witness)
        if include_witness:
            if enumerate_all:
                con_expl = [refine_contrastive_witness(e, sample) for e in con_expl]
            else:
                con_expl = refine_contrastive_witness(con_expl, sample)

        # For model-subject synthesis, prefer verified CXP witnesses.
        if self.stakeholder_type == 'model_subject' and include_witness:
            if enumerate_all:
                verified = [e for e in con_expl if _is_verified_cxp(e)]
                if verified:
                    con_expl = sorted(verified, key=_cxp_size)[:2]
                else:
                    print("[FRAME DEBUG][model_subject] No verified CXPs found in initial set; searching fallback CXPs.")
                    fallback = self.x.enumerate_cxps(sample, max_cxps=25, include_witness=True)
                    fallback = [refine_contrastive_witness(e, sample) for e in fallback]
                    verified = [e for e in fallback if _is_verified_cxp(e)]
                    if verified:
                        con_expl = sorted(verified, key=_cxp_size)[:2]
            else:
                if not _is_verified_cxp(con_expl):
                    fallback = self.x.enumerate_cxps(sample, max_cxps=25, include_witness=True)
                    fallback = [refine_contrastive_witness(e, sample) for e in fallback]
                    verified = [e for e in fallback if _is_verified_cxp(e)]
                    if verified:
                        con_expl = sorted(verified, key=_cxp_size)[0]
                        print("[FRAME DEBUG][model_subject] Fallback to verified enumerated CXP:", con_expl)
                    else:
                        print("[FRAME DEBUG][model_subject] No verified CXP found; using unverified single-shot CXP:", con_expl)
        expl = {'abd': abd_expl, 'con': con_expl}
        if self.stakeholder_type == 'developer':
            abd_axps = abd_expl if isinstance(abd_expl, list) and (not abd_expl or isinstance(abd_expl[0], (list, tuple, set))) else [abd_expl]
            expl['ffa'] = compute_ffa_from_axps(abd_axps)

        named_expl = map_named_features(expl)
        base = sample[0].copy() if sample.ndim > 1 else sample.copy()
        sample_map = {str(self.feature_names[i]): base[i] for i in range(len(self.feature_names))}
        pred = self._predict_single_row(base)

        def _fmt(v):
            if isinstance(v, (np.bool_, bool)):
                return str(bool(v))
            if isinstance(v, (np.integer, int)):
                return str(int(v))
            if isinstance(v, (np.floating, float)):
                return str(round(float(v), 5))
            return str(v)

        def _to_html(lines, exp_type):
            return [
                '<div class="xreason-{0}"><b>IF</b> {1}</div>'.format(
                    exp_type,
                    escape(line[3:]).replace(' THEN ', ' <b>THEN</b> ') if line.startswith('IF ') else escape(line)
                )
                for line in lines
            ]

        # map model output to user-facing label names when available
        effective_class_value_map = class_value_map if class_value_map is not None else self.class_value_map

        def _map_class_value(raw_value, fallback_text):
            if not effective_class_value_map:
                return str(fallback_text)
            candidates = [raw_value, str(raw_value)]
            if isinstance(raw_value, (np.integer, int)):
                candidates.append(int(raw_value))
                candidates.append(str(int(raw_value)))
            if isinstance(raw_value, (np.floating, float)):
                candidates.append(float(raw_value))
                candidates.append(str(float(raw_value)))
                if float(raw_value).is_integer():
                    iv = int(raw_value)
                    candidates.append(iv)
                    candidates.append(str(iv))
            for cand in candidates:
                if cand in effective_class_value_map:
                    return str(effective_class_value_map[cand])
            return str(fallback_text)

        pred_label = _fmt(pred)
        contrastive_label = None
        pred_idx = None
        classes = getattr(self.model, "classes_", None)
        if classes is not None:
            for i, c in enumerate(classes):
                if c == pred:
                    pred_idx = i
                    break
        if pred_idx is None and isinstance(pred, (np.integer, int)) and int(pred) < len(self.target_name):
            pred_idx = int(pred)
        if pred_idx is not None and pred_idx < len(self.target_name):
            pred_label = str(self.target_name[pred_idx])
        pred_label = _map_class_value(pred, pred_label)

        if pred_idx is not None and len(self.target_name) == 2:
            contrastive_label = str(self.target_name[1 - pred_idx])
            if classes is not None and len(classes) == 2:
                contrastive_label = _map_class_value(classes[1 - pred_idx], contrastive_label)
            else:
                contrastive_label = _map_class_value(1 - pred_idx, contrastive_label)

        def _abd_lines(axps_named):
            lines = []
            for axp in axps_named:
                cond = ' AND '.join(['{0} = {1}'.format(f, _fmt(sample_map.get(f, ''))) for f in axp])
                lines.append('IF {0} THEN label = {1}'.format(cond, pred_label))
            return lines

        def _con_lines(cxps_named):
            lines = []
            for cxp in cxps_named:
                if isinstance(cxp, dict):
                    feats = cxp.get('features', [])
                    vals = cxp.get('values', {})
                else:
                    feats = list(cxp)
                    vals = {}
                cond = ' AND '.join(['{0} = {1}'.format(f, _fmt(vals.get(f, sample_map.get(f, '')))) for f in feats])
                if contrastive_label is not None:
                    lines.append('IF {0} THEN label = {1}'.format(cond, contrastive_label))
                else:
                    lines.append('IF {0} THEN label != {1}'.format(cond, pred_label))
            return lines

        abd_items, con_items = [], []
        abd_part = named_expl.get('abd', []) if isinstance(named_expl, dict) else named_expl
        con_part = named_expl.get('con', []) if isinstance(named_expl, dict) else []
        if isinstance(abd_part, list) and abd_part and isinstance(abd_part[0], (list, tuple, set)):
            abd_items = abd_part
        elif abd_part:
            abd_items = [abd_part]
        if isinstance(con_part, list) and con_part and isinstance(con_part[0], (list, tuple, set, dict)):
            con_items = con_part
        elif con_part:
            con_items = [con_part]

        abd_strings = _abd_lines([a for a in abd_items if a is not None and a != []])
        con_strings = _con_lines([c for c in con_items if c is not None and c != []])

        explained_instance = 'IF {0} THEN label = {1}'.format(
            ' AND '.join(['{0} = {1}'.format(f, _fmt(sample_map[f])) for f in self.feature_names]),
            pred_label
        )

        result = {
            "explanations": named_expl,
            "explained_instance": explained_instance,
            "explanation_list": {"abd": abd_strings, "con": con_strings},
            "explanation_html": {"abd": _to_html(abd_strings, "abd"), "con": _to_html(con_strings, "con")}
        }

        # Optional FoX/VeriX-style FFA graphs and notebook widgets.
        ffa_scores = None
        if isinstance(named_expl, dict) and isinstance(named_expl.get("ffa"), dict):
            ffa_scores = named_expl["ffa"]
        elif isinstance(named_expl, dict) and isinstance(named_expl.get("abd"), list):
            abd_part = named_expl["abd"]
            if abd_part and isinstance(abd_part[0], list):
                ffa_scores = self._compute_ffa_from_named_axps(abd_part)
        elif isinstance(named_expl, dict) and isinstance(named_expl.get("axps"), list):
            axps_part = named_expl["axps"]
            if axps_part and isinstance(axps_part[0], list):
                ffa_scores = self._compute_ffa_from_named_axps(axps_part)
        
        # If no FFA scores computed yet, try from abductive items.
        if not ffa_scores and abd_items and abd_items[0] and isinstance(abd_items[0], (list, tuple)):
            ffa_scores = self._compute_ffa_from_named_axps(abd_items)

        # Generate FFA graphs only for developer stakeholder type
        if include_ffa_graph and ffa_scores and self.stakeholder_type == 'developer':
            graph_paths = self.save_ffa_graph(ffa_scores, out_dir=ffa_graph_dir)
            result["ffa_graph"] = graph_paths
            result["ffa"] = ffa_scores

        if in_jupyter:
            return self.render_in_jupyter_bundle(
                explained_instance=explained_instance,
                explanation_list=result["explanation_list"],
                pred=pred,
                pred_display=pred_label,
                prediction_label=prediction_label,
                ffa_graph=result.get("ffa_graph"),
                ffa_scores=ffa_scores
            )

        return result

    def _compute_ffa_from_named_axps(self, axps_named):
        # print("computing ffa")
        if not axps_named:
            return {}
        score = {}
        for axp in axps_named:
            if not axp:
                continue
            w = 1.0 / float(len(axp))
            for fname in axp:
                key = str(fname)
                score[key] = score.get(key, 0.0) + w
        n = float(len(axps_named))
        # print(n)
        return dict(sorted({k: (v / n) for k, v in score.items()}.items(), key=lambda kv: (-kv[1], kv[0])))

    def save_ffa_graph(self, f2imprt, out_dir="."):
        # print("saving ffa graph")
        if not f2imprt:
            return None
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("matplotlib is required for FFA graph generation") from e

        out_dir = str(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        all_path = os.path.join(out_dir, "ffa_all.png")
        top5_path = os.path.join(out_dir, "ffa_top5.png")

        # Only include features with non-zero influence
        filtered_scores = {k: v for k, v in f2imprt.items() if v != 0.0}

        sorted_items = sorted(filtered_scores.items(), key=lambda x: (abs(x[1]), x[0]))
        names_all = [k for k, v in sorted_items]
        values_all = [v for k, v in sorted_items]

        top5_items = sorted(filtered_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top5_items = sorted(top5_items, key=lambda x: (abs(x[1]), x[0]))
        names_top5 = [k for k, v in top5_items]
        values_top5 = [v for k, v in top5_items]
        # print("names_all", names_all)
        # print("values_all", values_all)
        # print("names_top5", names_top5)
        # print("values_top5", values_top5)
        #
        def save_feature_plot(names, values, filename):
            plt.rcParams['axes.linewidth'] = 2
            fig, ax = plt.subplots()
            # Dynamically size figure based on number of features
            height = max(4, len(names) * 0.25)
            fig.set_size_inches(6, height)
            for n, v in zip(names, values):
                if v > 0:
                    ax.barh(y=[n], width=[v], alpha=0.4, height=0.3, color=(0.2, 0.4, 0.6, 0.6))
                else:
                    ax.barh(y=[n], width=[v], alpha=0.8, height=0.3, color='orange')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.tick_params(axis='y', pad=3, labelsize=15)
            for h, (n, v) in enumerate(zip(names, values)):
                ax.text(v, h + .18, f'{v:.2f}', color='black',
                        horizontalalignment='left' if v > 0 else 'right', fontsize=10)
                ax.text(-.003 if v > 0 else .003, h - .05, n, color='black',
                        horizontalalignment='right' if v > 0 else 'left', fontsize=10)
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

        save_feature_plot(names_all, values_all, all_path)
        save_feature_plot(names_top5, values_top5, top5_path)
        with open(all_path, "rb") as f:
            all_data_url = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
        with open(top5_path, "rb") as f:
            top5_data_url = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
        return {
            "all": all_path,
            "top5": top5_path,
            "all_data_url": all_data_url,
            "top5_data_url": top5_data_url,
        }

    def _exp_mapping(self, if_else_text):
        mapped = []
        feature_value = if_else_text.split('THEN')[0]
        feature_value = feature_value.split('AND')
        feature_value = [word.strip("IF ") for word in feature_value]
        for fea_val_pair in feature_value:
            fea_val_pair = fea_val_pair.strip()
            if not fea_val_pair:
                continue
            if '!=' in fea_val_pair:
                fea_val = fea_val_pair.split('!=', 1)
                if len(fea_val) == 2:
                    mapped.append([fea_val[0].strip(), f"!= {fea_val[1].strip()}"])
            else:
                fea_val = fea_val_pair.split('=', 1)
                if len(fea_val) == 2:
                    mapped.append([fea_val[0].strip(), fea_val[1].strip()])
        label_side = if_else_text.split('THEN')[1].strip()
        if "!=" in label_side:
            label_value = label_side.split("!=", 1)
            mapped.append([label_value[0].strip(), label_value[1].strip()])
        elif "=" in label_side:
            label_value = label_side.split("=", 1)
            mapped.append([label_value[0].strip(), label_value[1].strip()])
        return mapped

    def render_in_jupyter_bundle(self, explained_instance, explanation_list, pred, pred_display=None, prediction_label="Prediction", ffa_graph=None, ffa_scores=None):
        try:
            import ipywidgets as widgets
            from IPython.display import display
            from foxplainer.html_string import HtmlString
        except Exception as e:
            raise RuntimeError("ipywidgets and foxplainer.html_string are required for in_jupyter rendering") from e

        def get_appropriate_n_expl(axps, cxps, stakeholder):
            def _exp_len(exp):
                if isinstance(exp, dict):
                    return len(exp.get('features', []))
                if isinstance(exp, str):
                    if 'IF ' in exp and ' THEN ' in exp:
                        cond = exp.split('IF ', 1)[1].split(' THEN ', 1)[0].strip()
                        return 0 if not cond else len([p for p in cond.split(' AND ') if p.strip()])
                    return 0
                try:
                    return len(exp)
                except TypeError:
                    return 0

            axps_sorted = sorted(axps, key=_exp_len)
            cxps_sorted = sorted(cxps, key=_exp_len)
            if (stakeholder == 'developer'):
                axps_selected = axps_sorted[:1]
                cxps_selected = cxps_sorted[:2]
            elif (stakeholder == 'decision_maker'):
                axps_selected = axps_sorted[:2]
                cxps_selected = cxps_sorted[:2]
            elif (stakeholder == 'model_subject'):
                axps_selected = axps_sorted[:1]
                cxps_selected = cxps_sorted[:2]
            else:
                axps_selected = axps_sorted[:1]
                cxps_selected = cxps_sorted[:1]
            return axps_selected, cxps_selected

        explained_instance_pairs = self._exp_mapping(explained_instance)
        axps = explanation_list['abd']
        cxps = explanation_list['con']
        axps_selected, cxps_selected = get_appropriate_n_expl(axps, cxps, self.stakeholder_type)
        abd_exp_html = ""
        con_exp_html = ""
        for exp in axps_selected:
            abd_exp_html += HtmlString(
                list_of_pair=self._exp_mapping(exp),
                exp_type="abd",
                prediction_label=prediction_label
            ).get_html()
        for exp in cxps_selected:
            con_exp_html += HtmlString(
                list_of_pair=self._exp_mapping(exp),
                exp_type="con",
                prediction_label=prediction_label
            ).get_html()

        explained_instance_html = HtmlString(
            list_of_pair=explained_instance_pairs,
            exp_type="abd",
            is_explained_instance=True,
            prediction_label=prediction_label
        ).get_html()

        tab_nest = widgets.Tab()
        accordion = widgets.Accordion(children=[tab_nest])
        accordion.set_title(index=0, title="Instance")

        children = [widgets.HTML(value=abd_exp_html or "<i>No abductive explanations</i>"),
                    widgets.HTML(value=con_exp_html or "<i>No contrastive explanations</i>")]
        tab_titles = ["Abductive Exp.", "Contrastive Exp."]

        if ffa_graph is not None:
            color = "rgba(237,34,14,255)" if str(pred) == "True" else "rgba(96,217,55,255)"
            # Debug: verify both paths are different
            print(f"FFA Graph all: {ffa_graph['all']}")
            print(f"FFA Graph top5: {ffa_graph['top5']}")
            all_src = ffa_graph.get("all_data_url", ffa_graph["all"])
            top5_src = ffa_graph.get("top5_data_url", ffa_graph["top5"])
            toggle_id = f"toggle_{id(ffa_graph)}"
            ffa_html = f'''
                        <div class="box">
                            <div class="inner-box"><text class="title"> Formal Feature Attribution</text></div>
                            <div class="bot-box-two" style="text-align: center;">
                            <input type="checkbox" id="{toggle_id}" style="display: none;">
                            <label for="{toggle_id}" class="toggle-label">
                                <span class="show-top5">Show Top 5 Features</span>
                                <span class="show-all">Show All Features</span>
                            </label>
                            <img src="{all_src}" class="img2">
                            <img src="{top5_src}" class="img1">
                            <style>
                            .img2 {{ display: none; }}
                            #{toggle_id}:checked ~ .img1 {{ display: none; }}
                            #{toggle_id}:checked ~ .img2 {{ display: inline; }}
                            .show-top5 {{ display: none; }}
                            #{toggle_id}:checked ~ .toggle-label .show-all {{ display: none; }}
                            #{toggle_id}:checked ~ .toggle-label .show-top5 {{ display: inline; }}
                            .toggle-label {{ display: inline-block; padding: 6px 12px; border: 1px solid #888; border-radius: 4px; background: #f5f5f5; cursor: pointer; user-select: none; }}
                            </style></div>
                            <div class="input-box"><div class="input-inner-box-grid-ffa" style="background-color: {color}; width: fit-content; display: inline-grid; grid-template-columns: max-content max-content; column-gap: 8px; padding: 4px 10px;">
                                <p class="general-text" style="white-space: nowrap;">{prediction_label}&nbsp;&nbsp;&nbsp;&nbsp;=</p>
                                <div class="input-container" style="width: auto; min-width: 48px; padding: 0 8px; white-space: nowrap;"><p class="input_text">{pred_display if pred_display is not None else pred}</p></div>
                            </div></div>
                        </div>'''
            children.append(widgets.HTML(value=ffa_html))
            tab_titles.append("Formal Feature Attribution")
        elif self.stakeholder_type == 'developer':
            children.append(widgets.HTML(value=(
                '<div style="padding:12px; border:1px solid #e0a800; border-radius:6px; '
                'background:#fff8e1; color:#7a5c00;">'
                '<b>⚠ Formal Feature Attribution unavailable</b><br>'
                'No explanations were found within the time limit. '
                'Try increasing <code>time_limit</code> and re-running.'
                '</div>'
            )))
            tab_titles.append("Formal Feature Attribution")

        if self.stakeholder_type == 'model_subject':
            synthesised_response = None
            if axps_selected and cxps_selected:
                from .gemini_synthesiser import synthesise_explanation
                print("[FRAME DEBUG][model_subject] AXP for synthesis:", axps_selected[0])
                print("[FRAME DEBUG][model_subject] CXPs for synthesis:", cxps_selected[:2])
                try:
                    synthesised_response = synthesise_explanation(
                        axp_string=axps_selected[0],
                        cxp_strings=cxps_selected[:2],
                        instance_explanation_string=explained_instance,
                        model='gemini-3.5-flash-lite'
                    )
                    print(synthesised_response)
                except (ValueError, RuntimeError, ImportError) as exc:
                    synthesised_response = "Synthesis unavailable: {0}".format(str(exc))

            if not synthesised_response:
                synthesised_response = "Synthesis unavailable: missing abductive or contrastive explanation."

            synthesis_html = '''
                <div class="box">
                    <div class="inner-box"><text class="title">Synthesised Explanation</text></div>
                    <div class="bot-box-two" style="padding: 12px 14px;">
                        <p style="margin:0; line-height:1.45;">{0}</p>
                    </div>
                </div>
            '''.format(escape(synthesised_response))
            children = [widgets.HTML(value=synthesis_html)]
            tab_titles = ["Synthesised Exp."]

        tab_nest.children = children
        for idx, title in enumerate(tab_titles):
            tab_nest.set_title(idx, title)
        return display(accordion)

    def validate(self, sample, expl):
        """
            Make an attempt to show that a given explanation is optimistic.
        """

        # there must exist an encoding
        self._ensure_smt_encoding()

        if 'v' not in dir(self):
            self.v = SMTValidator(self.enc, self.feature_names, self.num_class,
                    self)

        sample_arr = np.array(sample)
        self._validate_feature_bounds(sample_arr)

        # try to compute a counterexample
        return self.v.validate(sample_arr, expl)

    def transform(self, x):
        if(len(x) == 0):
            return x
        x = np.array(x)
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        if x.shape[1] != self.nb_features:
            model_n = int(getattr(self.model, "n_features_in_", self.nb_features))
            if x.shape[1] == model_n and self.nb_features != model_n:
                self.nb_features = model_n
            else:
                raise ValueError(
                    "Sample width mismatch: got {0} features, expected {1}.".format(x.shape[1], self.nb_features)
                )
        self._validate_feature_bounds(x)
        if (self.use_categorical):
            assert(self.encoder != [])
            tx = []
            for i in range(self.nb_features):
                if (i in self.categorical_features):
                    self.encoder[i].drop = None
                    tx_aux = self.encoder[i].transform(x[:,[i]])
                    if hasattr(tx_aux, 'toarray'):
                        tx_aux = tx_aux.toarray()
                    tx_aux = np.asarray(tx_aux)
                    tx.append(tx_aux)
                else:
                    tx.append(x[:,[i]])
            tx = np.hstack(tx)
            return tx
        else:
            return x

    def transform_inverse(self, x):
        if(len(x) == 0):
            return x
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        if (self.use_categorical):
            assert(self.encoder != [])
            inverse_x = []
            for i, xi in enumerate(x):
                inverse_xi = np.zeros(self.nb_features)
                for f in range(self.nb_features):
                    if f in self.categorical_features:
                        nb_values = len(self.categorical_names[f])
                        v = xi[:nb_values]
                        v = np.expand_dims(v, axis=0)
                        iv = self.encoder[f].inverse_transform(v)
                        inverse_xi[f] =iv
                        xi = xi[nb_values:]

                    else:
                        inverse_xi[f] = xi[0]
                        xi = xi[1:]
                inverse_x.append(inverse_xi)
            return inverse_x
        else:
            return x

    def transform_inverse_by_index(self, idx):
        if (idx in self.extended_feature_names):
            return self.extended_feature_names[idx]
        else:
            print("Warning there is no feature {} in the internal mapping".format(idx))
            return None

    def transform_by_value(self, feat_value_pair):
        if (feat_value_pair in self.extended_feature_names.values()):
            keys = (list(self.extended_feature_names.keys())[list( self.extended_feature_names.values()).index(feat_value_pair)])
            return keys
        else:
            print("Warning there is no value {} in the internal mapping".format(feat_value_pair))
            return None

    def mapping_features(self):
        self.extended_feature_names = {}
        self.extended_feature_names_as_array_strings = []
        counter = 0
        if (self.use_categorical):
            for i in range(self.nb_features):
                if (i in self.categorical_features):
                    for j, _ in enumerate(self.encoder[i].categories_[0]):
                        self.extended_feature_names.update({counter:  (self.feature_names[i], j)})
                        self.extended_feature_names_as_array_strings.append("f{}_{}".format(i,j)) # str(self.feature_names[i]), j))
                        counter = counter + 1
                else:
                    self.extended_feature_names.update({counter: (self.feature_names[i], None)})
                    self.extended_feature_names_as_array_strings.append("f{}".format(i)) #(self.feature_names[i])
                    counter = counter + 1
        else:
            for i in range(self.nb_features):
                self.extended_feature_names.update({counter: (self.feature_names[i], None)})
                self.extended_feature_names_as_array_strings.append("f{}".format(i))#(self.feature_names[i])
                counter = counter + 1

    def readable_sample(self, x):
        readable_x = []
        for i, v in enumerate(x):
            if (i in self.categorical_features):
                readable_x.append(self.categorical_names[i][int(v)])
            else:
                readable_x.append(v)
        return np.asarray(readable_x)

    def test_encoding_transformes(self):
        # test encoding

        if self.X_train is None or len(self.X_train) == 0:
            return

        X = self.X_train[[0],:]

        print("Sample of length", len(X[0])," : ", X)
        enc_X = self.transform(X)
        print("Encoded sample of length", len(enc_X[0])," : ", enc_X)
        inv_X = self.transform_inverse(enc_X)
        print("Back to sample", inv_X)
        print("Readable sample", self.readable_sample(inv_X[0]))
        assert((inv_X == X).all())

        if (self.verb > 1):
            for i in range(len(self.extended_feature_names)):
                print(i, self.transform_inverse_by_index(i))
            for key, value in self.extended_feature_names.items():
                print(value, self.transform_by_value(value))

    def transfomed_sample_info(self, i):
        print(enc.categories_)
