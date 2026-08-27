import random
import numpy as np
import onnx
import onnxruntime as ort
import base64
import importlib.util
from html import escape
from skimage.color import label2rgb
from matplotlib import pyplot as plt
from typing import Literal, Optional
import sys
from collections import defaultdict
from pathlib import Path
from foxplainer.pysat.examples.hitman import Hitman
import ipywidgets as widgets
from threading import Timer
from foxplainer.html_string import HtmlString

sys.path.insert(0, "Marabou")
"""
After installing Marabou, load it from maraboupy.
"""
from maraboupy import Marabou
from maraboupy import MarabouNetworkONNX
from maraboupy import MarabouCore, MarabouUtils


def _load_gemini_synthesiser():
    module_path = Path(__file__).resolve().parents[1] / "XReason" / "gemini_synthesiser.py"
    spec = importlib.util.spec_from_file_location("xreason_gemini_synthesiser", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.synthesise_explanation



class VeriX:
    """
    This is the VeriX class to take in an image and a neural network, and then output an explanation.
    """
    image = None
    keras_model = None
    mara_model = None
    traverse: str
    sensitivity = None
    datatype: str
    label: int
    inputVars = None
    outputVars = None
    epsilon: float
    """
    Marabou options: 'timeoutInSeconds' is the timeout parameter. 
    """
    options = Marabou.createOptions(numWorkers=16,
                                    timeoutInSeconds=300,
                                    verbosity=0,
                                    solveWithMILP=False)

    def __init__(self,
                 datatype,
                 model_path,
                 instance=None,
                 categorical_values_dict=None,
                 one_hot_groups=None,
                 feature_names=None,
                 time_limit=None,
                 in_jupyter=False,
                 plot_original=True,
                 prediction_label= "Prediction",
                 stakeholder_name='developer',
                 class_value_map=None):
        """
        To initialize the VeriX class.
        :param datatype: 'greyscale' or 'rgb' or 'tabular'
        :param instance: an instance of the data, tabular array or image array of shape (width, height, channel).
        :param model_path: the path to the neural network.
        :param plot_original: if True, then plot the original image.
        :param stakeholder_name: determines explanation granularity/mode.
               developer -> includes FFA tab for tabular data.
               decision_maker/model_subject -> single-shot only.
        :param categorical_values_dict: a dictionary to store the categorical columns and their potential values.
        :param one_hot_groups: optional list of mutually-exclusive feature groups
               (e.g. [['size_small', 'size_medium', 'size_large']]).
        """
        self.datatype: Literal["greyscale", "rgb", "tabular"] = datatype
        self.instance = None
        self.time_limit = time_limit
        self.stakeholder_name = str(stakeholder_name).strip().lower()
        """
        Load the onnx model.
        """
        self.model_path = model_path
        self.onnx_model = onnx.load(model_path)
        self.onnx_session = ort.InferenceSession(model_path)
        self.label = None
        self.pred = None
        self.categorical_columns = set(categorical_values_dict.keys()) if categorical_values_dict is not None else set()
        self.categorical_values = categorical_values_dict if categorical_values_dict is not None else {}
        self.feature_names = feature_names
        self.one_hot_groups = one_hot_groups if one_hot_groups is not None else []
        if categorical_values_dict is not None and feature_names is None:
            print("WARNING: categorical_values_dict given but no feature_names provided — "
                  "categorical constraints will not be applied correctly.")
        self._normalize_one_hot_groups()

        self.time_limit = time_limit
        self.traverse = "heuristic"  # default traversal order

        self.in_jupyter = in_jupyter
        self.tab_nest = widgets.Tab()
        self.accordion = widgets.Accordion(children=[self.tab_nest])  # ← add
        self.prediction_label = prediction_label
        self.class_value_map = class_value_map if class_value_map is not None else {}
        # self.inst_id = 0  # ← add
        self.abd_exp_html = ""  # ← add
        self.con_exp_html = ""  # ← add
        self.abd_con_exp_html = ""  # ← add
        self.instance_info_html = ""  # ← add
        self.explained_instance = ""  # ← add
        self.ffa_fig = None  # ← add (may already exist)
        self.ffa_all_img_src = ""
        self.ffa_top5_img_src = ""
        self.synthesis_exp_html = ""
        self.inputVars = None
        self.outputVars = None
        self.mara_model = None
        self._init_marabou_model()

        if instance is not None:
            self.set_instance(instance, plot_original=plot_original)

    def _init_marabou_model(self):
        """
        (Re)load the ONNX model into Marabou to ensure a clean constraint state
        for every new explained instance.
        """
        if self.onnx_model.graph.node[-1].op_type == "Softmax":
            mara_model_output = self.onnx_model.graph.node[-1].input
        else:
            mara_model_output = None
        self.mara_model = Marabou.read_onnx(filename=self.model_path, outputNames=mara_model_output)
        self.outputVars = self.mara_model.outputVars[0].flatten()

    def _reset_render_state(self):
        self.abd_exp_html = ""
        self.con_exp_html = ""
        self.abd_con_exp_html = ""
        self.instance_info_html = ""
        self.explained_instance = ""
        self.ffa_fig = None
        self.ffa_all_img_src = ""
        self.ffa_top5_img_src = ""
        self.synthesis_exp_html = ""

    def _require_instance(self):
        if self.instance is None:
            raise ValueError("No instance set. Call set_instance(instance) or explain(instance, ...) first.")

    def _uses_ffa_mode(self) -> bool:
        """
        Explanation mode is derived from stakeholder role.
        """
        return self.stakeholder_name == "developer"

    def _set_input_vars(self):
        if self.datatype in ["greyscale", "rgb"]:
            width, height = self.instance.shape[0], self.instance.shape[1]
            self.inputVars = np.arange(width * height)
        elif self.datatype == "tabular":
            self.inputVars = np.arange(self.instance.shape[0])
        else:
            raise ValueError(f"Unsupported datatype: {self.datatype}")

    def _predict_for_instance(self, instance):
        onnx_input_data = np.expand_dims(instance, axis=0)
        prediction = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: onnx_input_data})
        prediction = np.asarray(prediction[0])
        return prediction.argmax()

    def set_instance(self, instance, plot_original=True):
        self.instance = np.asarray(instance)
        if self.datatype == "tabular" and self.feature_names is not None and len(self.feature_names) != self.instance.shape[0]:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) does not match "
                f"tabular instance width ({self.instance.shape[0]})."
            )
        if self.datatype == "tabular" and self.feature_names is None:
            self.feature_names = [f"f{i}" for i in range(self.instance.shape[0])]
        self.label = self._predict_for_instance(self.instance)
        self.pred = self.label
        self._set_input_vars()
        self._reset_render_state()
        if self.datatype == "tabular" and self.one_hot_groups:
            self._validate_one_hot_instance(self.instance)

        if plot_original and (self.datatype in ["greyscale", "rgb"]):
            save_figure(
                image=self.instance,
                path=f"original-predicted-as-{self.label}.png",
                cmap="gray" if self.datatype == 'greyscale' else None
            )

    def explain(self, instance, epsilon, plot_explanation=True, plot_counterfactual=False, plot_timeout=False,
                plot_original=False, in_jupyter=None, prediction_label=None, class_value_map=None):
        """
        Reuse a single VeriX object across many samples.
        :param in_jupyter: override self.in_jupyter for this call (True/False), or None to use the instance setting.
        :param prediction_label: override self.prediction_label for this call, or None to use the instance setting.
        """
        self.set_instance(instance, plot_original=plot_original)
        return self.get_explanation(
            epsilon=epsilon,
            plot_explanation=plot_explanation,
            plot_counterfactual=plot_counterfactual,
            plot_timeout=plot_timeout,
            in_jupyter=in_jupyter,
            prediction_label=prediction_label,
            class_value_map=class_value_map,
        )

    def traversal_order(self,
                        traverse="heuristic",
                        plot_sensitivity=True,
                        seed=0):
        """
        To compute the traversal order of checking all the pixels in the image.
        :param traverse: 'heuristic' (by default) or 'random'.
        :param plot_sensitivity: if True, plot the sensitivity map.
        :param seed: if traverse by 'random', then set a random seed.
        :return: an updated inputVars that contains the traversal order.
        """
        self.traverse = traverse
        if self.traverse == "heuristic":
            if self.datatype == "greyscale" or self.datatype == "rgb":
                width, height, channel = self.instance.shape[0], self.instance.shape[1], self.instance.shape[2]
                temp = self.instance.reshape(width * height, channel)
                instance_batch = np.kron(np.ones(shape=(width * height, 1, 1), dtype=temp.dtype), temp)
                instance_batch_manip = instance_batch.copy()
                for i in range(width * height):
                    """
                    Different ways to compute sensitivity: use pixel reversal for MNIST and deletion for GTSRB.
                    """
                    if self.datatype == "greyscale":
                        instance_batch_manip[i][i][:] = 1 - instance_batch_manip[i][i][:]
                    elif self.datatype == "rgb":
                        instance_batch_manip[i][i][:] = 0
                    else:
                        print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
                instance_batch = instance_batch.reshape((width * height, width, height, channel))
                predictions = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: instance_batch})
                predictions = np.asarray(predictions[0])
                instance_batch_manip = instance_batch_manip.reshape((width * height, width, height, channel))
                predictions_manip = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: instance_batch_manip})
                predictions_manip = np.asarray(predictions_manip[0])
                difference = predictions - predictions_manip
                features = difference[:, self.label]
                sorted_index = features.argsort()
                self.inputVars = sorted_index
                self.sensitivity = features.reshape(width, height)
                if plot_sensitivity and self.datatype in ["greyscale", "rgb"]:
                    filename = f"{self.datatype}-sensitivity-{self.traverse}.png"
                    save_figure(image=self.sensitivity, path=filename)
                    if self.in_jupyter:
                        from IPython.display import display, Image
                        display(Image(filename=filename))
            elif self.datatype == "tabular":
                random.seed(seed)
                random.shuffle(self.inputVars)
                """ TODO: 
                    For heuristic for tabular implement something like SHAP here for an approximation of

                """
        elif self.traverse == "random" or self.datatype =="tabular":
            random.seed(seed)
            random.shuffle(self.inputVars)
        else:
            print("Traversal not supported: try 'heuristic' or 'random'.")

    def _add_value_disjunction(self, var_index, allowed_values):
        """
        Add disjunction (x == v1) OR (x == v2) ... using Marabou Equation objects.
        """
        disjuncts = []
        for v in allowed_values:
            eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
            eq.addAddend(1, int(var_index))
            eq.setScalar(float(v))
            disjuncts.append([eq])
        if disjuncts:
            self.mara_model.addDisjunctionConstraint(disjuncts)

    def _normalize_feature_ref(self, ref):
        if isinstance(ref, int):
            return ref
        if self.feature_names is None:
            raise ValueError("feature_names are required when one_hot_groups uses names")
        name_to_idx = {str(name): i for i, name in enumerate(self.feature_names)}
        if ref in name_to_idx:
            return name_to_idx[ref]
        if isinstance(ref, str) and ref.lower() in {k.lower(): v for k, v in name_to_idx.items()}:
            lookup = {k.lower(): v for k, v in name_to_idx.items()}
            return lookup[ref.lower()]
        raise ValueError(f"Unknown feature reference in one_hot_groups: {ref}")

    def _normalize_one_hot_groups(self):
        normalized = []
        for group in self.one_hot_groups or []:
            if not group:
                continue
            idxs = sorted({self._normalize_feature_ref(ref) for ref in group})
            if len(idxs) > 1:
                normalized.append(idxs)
        self.one_hot_groups = normalized

    def _validate_one_hot_instance(self, instance):
        for group in self.one_hot_groups:
            active = [i for i in group if float(instance[i]) == 1.0]
            if len(active) > 1:
                names = [self.feature_names[i] for i in active] if self.feature_names is not None else active
                raise ValueError(f"One-hot group has multiple active features: {names}")

    def _add_one_hot_group_constraints(self, group, instance=None):
        """
        Enforce exactly one active feature in a one-hot group.
        """
        if instance is not None:
            self._validate_one_hot_instance(instance)

        # at least one must be 1
        one_disjuncts = []
        for i in group:
            eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
            eq.addAddend(1, int(i))
            eq.setScalar(1.0)
            one_disjuncts.append([eq])
        if one_disjuncts:
            self.mara_model.addDisjunctionConstraint(one_disjuncts)

        # no pair can both be 1
        for idx, i in enumerate(group):
            for j in group[idx + 1:]:
                pair_disjuncts = []
                for v in (i, j):
                    eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
                    eq.addAddend(1, int(v))
                    eq.setScalar(0.0)
                    pair_disjuncts.append([eq])
                self.mara_model.addDisjunctionConstraint(pair_disjuncts)

    def _set_categorical_bounds_and_disjunction(self, var_index, all_values, current_value):
        """
        Categorical variables still need finite bounds in Marabou before adding
        disjunction constraints, otherwise it may report infinite bounds.
        """
        values = [float(v) for v in all_values]
        self.mara_model.setLowerBound(var_index, min(values))
        self.mara_model.setUpperBound(var_index, max(values))
        allowed_values = [v for v in values if v != float(current_value)]
        self._add_value_disjunction(var_index, allowed_values)

    def get_explanation(self,
                        epsilon,
                        plot_explanation=True,
                        plot_counterfactual=False,
                        plot_timeout=False,
                        in_jupyter=None,
                        prediction_label=None,
                        class_value_map=None):
        """
        To compute the explanation for the model and the neural network.
        :param epsilon: the perturbation magnitude.
        :param plot_explanation: if True, plot the explanation.
        :param plot_counterfactual: if True, plot the counterfactual(s).
        :param plot_timeout: if True, plot the timeout pixel(s).
        :param in_jupyter: override self.in_jupyter for this call (True/False), or None to use the instance setting.
        :param prediction_label: override self.prediction_label for this call, or None to use the instance setting.
        :return: an explanation, and possible counterfactual(s).
        """
        use_jupyter = in_jupyter if in_jupyter is not None else self.in_jupyter
        use_prediction_label = prediction_label if prediction_label is not None else self.prediction_label
        use_class_value_map = class_value_map if class_value_map is not None else self.class_value_map
        self._require_instance()
        self._init_marabou_model()
        self._reset_render_state()

        unsat_set = []
        sat_set = []
        timeout_set = []
        
        # Always compute single-shot base explanation first.
        if self.datatype in {"greyscale", "rgb", "tabular"}:
            if(self.datatype == "greyscale" or self.datatype=="rgb"):
                width, height, channel = self.instance.shape[0], self.instance.shape[1], self.instance.shape[2]
                instance = self.instance.reshape(width * height, channel)
            else:
                instance = self.instance

            for feat in self.inputVars:
                for i in self.inputVars:
                    """
                    Set constraints on the input variables.
                    """
                    if i == feat or i in unsat_set:
                        """
                        Set allowable perturbations on the current pixel/feature and the irrelevant pixels/features.
                        """
                        if self.datatype == "greyscale":
                            self.mara_model.setLowerBound(i, max(0, instance[i][:] - epsilon))
                            self.mara_model.setUpperBound(i, min(1, instance[i][:] + epsilon))
                        elif self.datatype == "rgb":
                            self.mara_model.setLowerBound(3 * i, max(0, instance[i][0] - epsilon))
                            self.mara_model.setUpperBound(3 * i, min(1, instance[i][0] + epsilon))
                            self.mara_model.setLowerBound(3 * i + 1, max(0, instance[i][1] - epsilon))
                            self.mara_model.setUpperBound(3 * i + 1, min(1, instance[i][1] + epsilon))
                            self.mara_model.setLowerBound(3 * i + 2, max(0, instance[i][2] - epsilon))
                            self.mara_model.setUpperBound(3 * i + 2, min(1, instance[i][2] + epsilon))
                        elif self.datatype == "tabular":
                            # We can implement more natural bounds for them in the future/allow user to provide.
                            col = self.feature_names[i]
                            if col not in self.categorical_columns:
                                if epsilon is None:
                                    INF = 1e9
                                    self.mara_model.setLowerBound(i, -INF)
                                    self.mara_model.setUpperBound(i, INF)
                                else: #might need to make sure this makes sense later
                                    self.mara_model.setLowerBound(i, instance[i]-epsilon)
                                    self.mara_model.setUpperBound(i, instance[i]+epsilon)
                            else:
                                self._set_categorical_bounds_and_disjunction(
                                    i,
                                    self.categorical_values[col],
                                    instance[i]
                                )
                        # else:
                        #     print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
                    else:
                        """
                        Make sure the other pixels/features are fixed.
                        """
                        if self.datatype == "greyscale":
                            self.mara_model.setLowerBound(i, instance[i][:])
                            self.mara_model.setUpperBound(i, instance[i][:])
                        elif self.datatype == "rgb":
                            self.mara_model.setLowerBound(3 * i, instance[i][0])
                            self.mara_model.setUpperBound(3 * i, instance[i][0])
                            self.mara_model.setLowerBound(3 * i + 1, instance[i][1])
                            self.mara_model.setUpperBound(3 * i + 1, instance[i][1])
                            self.mara_model.setLowerBound(3 * i + 2, instance[i][2])
                            self.mara_model.setUpperBound(3 * i + 2, instance[i][2])
                        elif self.datatype == "tabular":
                            self.mara_model.setLowerBound(i, instance[i])
                            self.mara_model.setUpperBound(i,instance[i])

                        # else:
                        #     print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
            if self.datatype == "tabular" and self.one_hot_groups:
                for group in self.one_hot_groups:
                    self._add_one_hot_group_constraints(group, instance=instance)
            exit_code = None
            for j in range(len(self.outputVars)):
                """
                Set constraints on the output variables.
                """
                if j != self.label:
                    self.mara_model.addInequality([self.outputVars[self.label], self.outputVars[j]],
                                                  [1, -1], -1e-6,
                                                  isProperty=True)
                    exit_code, vals, stats = self.mara_model.solve(options=self.options, verbose=False)
                    """
                    additionalEquList.clear() is to clear the output constraints.
                    """
                    self.mara_model.additionalEquList.clear()
                    if exit_code == 'sat' or exit_code == 'TIMEOUT':
                        break
                    elif exit_code == 'unsat':
                        continue
            """
            clearProperty() is to clear both input and output constraints.
            Disjunctions (categorical/one-hot) must also be cleared manually,
            as clearProperty() does not touch disjunctionList.
            """
            self.mara_model.clearProperty()
            self.mara_model.disjunctionList.clear()
            """
            If unsat, put the pixel into the irrelevant set; 
            if timeout, into the timeout set; 
            if sat, into the explanation.
            """
            if exit_code == 'unsat':
                unsat_set.append(feat)
            elif exit_code == 'TIMEOUT':
                timeout_set.append(feat)
            elif exit_code == 'sat':
                sat_set.append(feat)
                if plot_counterfactual:
                    counterfactual = [vals.get(i) for i in self.mara_model.inputVars[0].flatten()]
                    counterfactual = np.asarray(counterfactual).reshape(self.instance.shape)
                    prediction = [vals.get(i) for i in self.outputVars]
                    prediction = np.asarray(prediction).argmax()
                    if self.datatype in ["greyscale", "rgb"]:
                        save_figure(image=counterfactual,
                                    path="counterfactual-at-pixel-%d-predicted-as-%d.png" % (feat, prediction),
                                    cmap="gray" if self.datatype == 'greyscale' else None)
                    else:
                        print(f"Counterfactual for tabular data at feature {feat} with value {counterfactual[feat]} predicted as {prediction}.")

            if plot_explanation:
                mask = np.zeros(self.inputVars.shape).astype(bool)
                mask[sat_set] = True
                mask[timeout_set] = True
                plot_shape = self.instance.shape[0:2] if self.datatype == "greyscale" else self.instance.shape
                save_figure(image=label2rgb(mask.reshape(self.instance.shape[0:2]),
                                            self.instance.reshape(plot_shape),
                                            colors=[[0, 1, 0]] if self.traverse == 'heuristic' else [[1, 0, 0]],
                                            bg_label=0,
                                            saturation=1),
                            path="explanation-%d.png" % (len(sat_set) + len(timeout_set)))
                if use_jupyter and self.datatype in ["rgb", "greyscale"]:
                    from IPython.display import display, Image

                    display(Image(
                        filename=f"explanation-{len(sat_set) + len(timeout_set)}.png"
                    ))
            if plot_timeout:

                mask = np.zeros(self.inputVars.shape).astype(bool)
                mask[timeout_set] = True
                if self.datatype == "greyscale" or self.datatype == "rgb":
                    plot_shape = self.instance.shape[0:2] if self.datatype == "greyscale" else self.instance.shape
                    save_figure(image=label2rgb(mask.reshape(self.instance.shape[0:2]),
                                                self.instance.reshape(plot_shape),
                                                colors=[[0, 1, 0]] if self.traverse == 'heuristic' else [[1, 0, 0]],
                                                bg_label=0,
                                                saturation=1),
                                path="timeout-%d.png" % len(timeout_set))
                else:
                    for i in timeout_set:
                        print(f"Feature {i}: value = {self.instance[i]}")
            if self.datatype == 'tabular':
               print("Relevant features (sat):", sat_set)
               print("Irrelevant features (unsat):", unsat_set)
               print("Timeout features:", timeout_set)

        # FFA enumeration only for developer-facing tabular explanations
        if self._uses_ffa_mode() and self.datatype == "tabular":
            # For FFA, we already have sat_set from the single-shot computation above
            # Set up constraints for feature perturbation
            instance = self.instance
            for feat in self.inputVars:
                for i in self.inputVars:
                    if i == feat or i in unsat_set:
                        col = self.feature_names[i]
                        if col not in self.categorical_columns:
                            if epsilon is None:
                                INF = 1e9
                                self.mara_model.setLowerBound(i, -INF)
                                self.mara_model.setUpperBound(i, INF)
                            else:  # might need to make sure this makes sense later
                                self.mara_model.setLowerBound(i, instance[i] - epsilon)
                                self.mara_model.setUpperBound(i, instance[i] + epsilon)
                        else:
                            self._set_categorical_bounds_and_disjunction(
                                i,
                                self.categorical_values[col],
                                instance[i]
                            )
                    else:
                        self.mara_model.setLowerBound(i, instance[i])
                        self.mara_model.setUpperBound(i, instance[i])
            if self.datatype == "tabular" and self.one_hot_groups:
                for group in self.one_hot_groups:
                    self._add_one_hot_group_constraints(group, instance=instance)
            
            # Get single-shot explanation and enumerate all AXPs
            axps_single = sorted(sat_set)
            # Enumerate all AXPs (including and beyond the single-shot)
            axps, cxps = self.enumerate_all_axps_tabular_fast(epsilon=epsilon)
            
            # Ensure single-shot is first if not already discovered
            if axps_single not in axps:
                axps.insert(0, axps_single)

        # Convert AXPs/CXPs to explanation strings (for both single-shot and FFA modes)
        if self.datatype == "tabular":
            if not self._uses_ffa_mode():
                # For single-shot, only use the sat_set as a single AXP if it's non-trivial
                if sat_set:
                    axps = [sorted(sat_set)]
                else:
                    axps = []
                cxps = []
                 
                if not axps:
                    # If single-shot yields no relevant features (trivial "IF TRUE" explanation),
                    # run a short bounded enumeration to find explanations based on stakeholder.
                    short_budget = 30 if self.time_limit is None else min(float(self.time_limit), 30.0)
                    
                    # Determine how many AXPs to find based on stakeholder type
                    num_axps_target = 2 if self.stakeholder_name == 'decision_maker' else 1
                    
                    fb_axps, fb_cxps = self.enumerate_all_axps_tabular_fast(
                        epsilon=epsilon,
                        max_seconds=short_budget,
                        num_axps=num_axps_target
                    )
                    # Filter out empty/trivial AXPs (those that would be "IF TRUE")
                    # and keep non-empty ones up to the target count
                    non_empty_axps = [axp for axp in fb_axps if axp]
                    axps = non_empty_axps[:num_axps_target] if non_empty_axps else []
                    if fb_cxps:
                        cxps = fb_cxps

            def get_appropriate_n_expl(axps, cxps, stakeholder):
                # Filter out trivial/empty AXPs (those that would be "IF TRUE")
                axps_nontrivial = [axp for axp in axps if axp]
                 
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
 
                axps_sorted = sorted(axps_nontrivial, key=_exp_len)
                cxps_sorted = sorted(cxps, key=_exp_len)
                if(stakeholder == 'developer'):
                    axps_selected = axps_sorted[:1]
                    cxps_selected = cxps_sorted[:2]
                elif(stakeholder == 'decision_maker'):
                    axps_selected = axps_sorted[:2]
                    cxps_selected = cxps_sorted[:2]
                elif(stakeholder == 'model_subject'):
                    axps_selected = axps_sorted[:1]
                    cxps_selected = cxps_sorted[:2]
                else:
                    axps_selected = axps_sorted[:1]
                    cxps_selected = cxps_sorted[:1]
                return axps_selected, cxps_selected

            axps_selected, cxps_selected = get_appropriate_n_expl(axps, cxps, stakeholder=self.stakeholder_name)
            expl_strings = self.axps_to_fox_abd_strings(axps_selected, cxps_selected, class_value_map=use_class_value_map)
            self.synthesis_exp_html = ""
            if self.stakeholder_name == 'model_subject' and axps_selected and cxps_selected:
                instance_context = self._build_explained_instance_string(class_value_map=use_class_value_map)
                try:
                    synthesise_explanation = _load_gemini_synthesiser()
                    print("[FRAME DEBUG][model_subject] AXP for synthesis:", expl_strings['abd'][0])
                    print("[FRAME DEBUG][model_subject] CXPs for synthesis:", expl_strings['con'][:2])
                    synthesised_response = synthesise_explanation(
                        axp_string=expl_strings['abd'][0],
                        cxp_strings=expl_strings['con'][:2],
                        instance_explanation_string=instance_context,
                        model='gemini-3.5-flash-lite'
                    )
                    self.synthesis_exp_html = '''
                        <div class="box">
                            <div class="inner-box"><text class="title">Synthesised Explanation</text></div>
                            <div class="bot-box-two" style="padding: 12px 14px;">
                                <p style="margin:0; line-height:1.45;">{0}</p>
                            </div>
                        </div>
                    '''.format(escape(synthesised_response))
                except (ValueError, RuntimeError, ImportError) as exc:
                    self.synthesis_exp_html = '''
                        <div style="padding:12px; border:1px solid #e0a800; border-radius:6px;
                                    background:#fff8e1; color:#7a5c00;">
                            <b>⚠ Synthesised explanation unavailable</b><br>
                            {0}
                        </div>
                    '''.format(escape(str(exc)))
            print(f"sat_set: {sat_set}")
            # print(f"axps: {axps}")
            print(f"expl_strings: {expl_strings}")
            _saved_label = self.prediction_label
            self.prediction_label = use_prediction_label
            self.exp_to_html(exp_list=expl_strings['abd'], exp_type='abd')
            self.exp_to_html(exp_list=expl_strings['con'], exp_type='con')
            self.prediction_label = _saved_label

            ffa_expl_strings = self.axps_to_fox_abd_strings(axps, [], class_value_map=use_class_value_map)
            
            if self._uses_ffa_mode():
                # FFA only computed for full enumeration mode
                ffa = self.ffa_like_fox(ffa_expl_strings)
                print(f"FFA result: {ffa}")

                print(f"axps count: {len(axps)}, ffa empty: {ffa == {} }")
                if ffa != {}:
                    self.save_ffa_graph(ffa)
                    title = ""
                    exp_type_full = "Formal Feature Attribution"

                    # if str(self.pred) == "True":
                    #     color = "rgba(237,34,14,255)"
                    # else:
                    #     color = "rgba(96,217,55,255)"
                    color = "rgba(96,217,55,255)"
                    equal_sign = "&nbsp;&nbsp;&nbsp;&nbsp;="
                    label_title = use_prediction_label
                    pred_display = self._format_class_value(self.pred, class_value_map=use_class_value_map)
                    ffa_html = f'''
                                <div class="box">
                                    <div class="inner-box">
                                        <text class="title">{title} {exp_type_full}</text>
                                    </div>
                                    <div class="bot-box-two" style="text-align: center;">
                                    <input type="checkbox" id="toggle">
                                    <label for="toggle" class="toggle-label">
                                        <span class="show-all">Show Top 5 Features</span>
                                        <span class="show-top5">Show All Features</span>
                                    </label>
    
                                    <img src="{self.ffa_all_img_src}" class="img1">
                                    <img src="{self.ffa_top5_img_src}" class="img2">
    
                                    <style>
                                    /* Images */
                                    .img2 {{
                                        display: none;
                                    }}
    
                                    #toggle:checked ~ .img1 {{
                                        display: none;
                                    }}
    
                                    #toggle:checked ~ .img2 {{
                                        display: inline;
                                    }}
    
                                    /* Label text */
                                    .show-top5 {{
                                        display: none;
                                    }}
    
                                    #toggle:checked + .toggle-label .show-all {{
                                        display: none;
                                    }}
    
                                    #toggle:checked + .toggle-label .show-top5 {{
                                        display: inline;
                                    }}
    
                                    /* Optional styling */
                                    .toggle-label {{
                                        display: inline-block;
                                        padding: 6px 12px;
                                        border: 1px solid #888;
                                        border-radius: 4px;
                                        background: #f5f5f5;
                                        cursor: pointer;
                                        user-select: none;
                                    }}
                                    </style>
                                    </div>
    
                                    <div class="input-box">
                                        <div class="input-inner-box-grid-ffa" style="background-color: {color}; width: fit-content; display: inline-grid; grid-template-columns: max-content max-content; column-gap: 8px; padding: 4px 10px;">
                                            <p class="general-text" style="white-space: nowrap;">{label_title}{equal_sign}</p>
                                            <div class="input-container" style="width: auto; min-width: 48px; padding: 0 8px; white-space: nowrap;">
                                                <p class="input_text">{pred_display}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                '''
                    self.ffa_fig = widgets.HTML(value=ffa_html)
            
            # Display in Jupyter if enabled
            print(f"DEBUG: in_jupyter={use_jupyter}, datatype={self.datatype}")
            if use_jupyter:
                print("DEBUG: Calling show_in_jupyter")
                self.show_in_jupyter(show_both_exp=self._uses_ffa_mode())

    # def extract_mus(self, start_from=None):
    #     """
    #         Compute any subset-minimal explanation.
    #     """
    #     self.nsat, self.nunsat = 0, 0
    #     self.stimes, self.utimes = [], []
    #     vtaut = self.enc.newVar('Tautology')
    #
    #     def _do_linear(core):
    #         """
    #             Do linear search.
    #         """
    #
    #         def _assump_needed(a):
    #             if len(to_test) > 1:
    #                 to_test.remove(a)
    #                 self.calls += 1
    #                 sat = self.slv.solve(assumptions=[vtaut] + sorted(to_test))
    #                 if not sat:
    #                     self.nunsat += 1
    #                     return False
    #                 to_test.add(a)
    #                 self.nsat += 1
    #                 return True
    #             else:
    #                 return True
    #
    #         to_test = set(core)
    #         return list(filter(lambda a: _assump_needed(a), core))
    #
    #     if start_from is None:
    #         # this call must be unsatisfiable!
    #         assert self.slv.solve(assumptions=[vtaut] + self.assums) == False
    #     else:
    #         assert self.slv.solve(assumptions=[vtaut] + start_from) == False
    #     # this is our MUS over-approximation
    #     core = self.slv.get_core()
    #     core = list(filter(lambda l: l != vtaut, core))
    #     expl = _do_linear(core)
    #     return expl
    #
    #
    # def mhs_mus_enumeration(self):
    #     """
    #         Enumerate subset- and cardinality-minimal explanations.
    #     """
    #     # result
    #     self.expls = []
    #     # just in case, let's save dual (contrastive) explanations
    #     self.duals = []
    #     vtaut = self.enc.newVar('Tautology')
    #
    #     timed_out = [False]
    #     timer = None
    #
    #     def interrupt():
    #         timed_out[0] = True
    #
    #     try:
    #         start_time = time.time()
    #         with Hitman(bootstrap_with=[self.assums], htype='sorted' if self.options.smallest else 'lbx') as hitman:
    #             # computing unit-size MCSes
    #             for i, hypo in enumerate(self.assums):
    #                 self.calls += 1
    #                 if self.slv.solve(assumptions=[vtaut] + self.assums[:i] + self.assums[(i + 1):]):
    #                     hitman.hit([hypo])
    #                     self.duals.append([hypo])
    #
    #                     if self.options.xtype in ('con', 'contrastive') and self.options.xnum not in (-1, 'all'):
    #                         return
    #
    #                 else:
    #                     pass
    #             # main loop
    #             end_time = time.time()
    #             print(f"time taken for unit-size MCSes: {end_time - start_time:.2f} seconds")
    #             if self.time_limit is not None:
    #                 timer = Timer(self.time_limit, interrupt)
    #                 timer.start()
    #             iters = 0
    #             while True:
    #                 if timed_out[0]:
    #                     return
    #                 hset = hitman.get()
    #                 iters += 1
    #                 if hset is None:
    #                     break
    #                 self.calls += 1
    #                 if self.slv.solve(assumptions=[vtaut] + hset):
    #                     to_hit = []
    #                     satisfied, unsatisfied = [], []
    #                     removed = list(set(self.assums).difference(set(hset)))
    #                     model = self.slv.get_model()
    #                     for h in removed:
    #                         if model[abs(h) - 1] != h:
    #                             unsatisfied.append(h)
    #                         else:
    #                             hset.append(h)
    #                     # computing an MCS (expensive)
    #                     for h in unsatisfied:
    #                         self.calls += 1
    #                         if self.slv.solve(assumptions=[vtaut] + hset + [h]):
    #                             hset.append(h)
    #                         else:
    #                             to_hit.append(h)
    #                     hitman.hit(to_hit)
    #                     self.duals.append(to_hit)
    #
    #                     if self.options.xtype in ('con', 'contrastive') and self.options.xnum not in (-1, 'all'):
    #                         break
    #                 # AXP?
    #                 else:
    #                     self.expls.append(hset)
    #                     if len(self.expls) != self.options.xnum:
    #                         hitman.block(hset)
    #                     else:
    #                         break
    #     finally:
    #         if self.options.time_limit is not None:
    #             if timer is not None:
    #                 timer.cancel()
    #         print("exiting enumeration")

    def _query_status_tabular(self, fixed_lits, epsilon):
        """
        fixed_lits: set of 1-based feature literals that are fixed.
        Returns one of: 'sufficient', 'insufficient', 'timeout'
        """
        assert self.datatype == "tabular"
        n = self.instance.shape[0]
        fixed = set(fixed_lits)
        input_var_ids = self.mara_model.inputVars[0].flatten()  # var id per 0-based feature

        # Set input bounds
        for lit in range(1, n + 1):
            i = lit - 1
            xi = float(self.instance[i])
            if lit in fixed:
                self.mara_model.setLowerBound(i, xi)
                self.mara_model.setUpperBound(i, xi)
            else:
                col = self.feature_names[i]
                if col in self.categorical_columns:
                    # Constrain to the other valid categorical levels, not a continuous ball.
                    all_values = [float(v) for v in self.categorical_values[col]]
                    self.mara_model.setLowerBound(i, min(all_values))
                    self.mara_model.setUpperBound(i, max(all_values))
                    allowed_values = [v for v in all_values if v != xi]
                    if not allowed_values:
                        # No alternate levels to try; effectively fixed.
                        self.mara_model.setLowerBound(i, xi)
                        self.mara_model.setUpperBound(i, xi)
                    else:
                        self._add_value_disjunction(i, allowed_values)
                else:
                    if epsilon is None:
                        INF = 1e9
                        self.mara_model.setLowerBound(i, -INF)
                        self.mara_model.setUpperBound(i, INF)
                    else:
                        self.mara_model.setLowerBound(i, xi - epsilon)
                        self.mara_model.setUpperBound(i, xi + epsilon)

        # Check if any competing class can beat current label
        exit_code = None
        for j in range(len(self.outputVars)):
            if j == self.label:
                continue
            self.mara_model.addInequality(
                [self.outputVars[self.label], self.outputVars[j]],
                [1, -1],
                -1e-6,
                isProperty=True
            )
            exit_code, vals, _ = self.mara_model.solve(options=self.options, verbose=False)
            self.mara_model.additionalEquList.clear()

            if exit_code == "sat":
                witness_values = {}
                for lit in range(1, n + 1):
                    if lit not in fixed:  # only the free features matter
                        var_id = input_var_ids[lit - 1]
                        witness_values[lit] = vals.get(var_id)
                self.mara_model.clearProperty()
                self.mara_model.disjunctionList.clear()
                return "insufficient", {"values": witness_values, "target_class": j}
            if exit_code == "TIMEOUT":
                self.mara_model.clearProperty()
                self.mara_model.disjunctionList.clear()
                return "timeout", None

        self.mara_model.clearProperty()
        self.mara_model.disjunctionList.clear()
        return "sufficient", None
    #
    # def _extract_axp_tabular(self, seed_lits, epsilon, cache):
    #     """
    #     seed_lits must be sufficient. Returns subset-minimal sufficient set.
    #     """
    #     print("found axp")
    #     axp = set(seed_lits)
    #     for lit in list(seed_lits):
    #         trial = set(axp)
    #         trial.remove(lit)
    #         key = frozenset(trial)
    #         status = cache.get(key)
    #         if status is None:
    #             status = self._query_status_tabular(trial, epsilon)
    #             cache[key] = status
    #         if status == "sufficient":
    #             axp.remove(lit)
    #         elif status == "timeout":
    #             return None  # unknown
    #     return axp
    #
    # def _extract_cxp_tabular(self, seed_lits, universe_lits, epsilon, cache):
    #     """
    #     seed_lits is insufficient. Returns subset-minimal CXP (free features).
    #     """
    #     print("found cxp")
    #     free = set(universe_lits).difference(seed_lits)
    #     cxp = set(free)
    #
    #     for lit in list(free):
    #         trial_cxp = set(cxp)
    #         trial_cxp.remove(lit)
    #         trial_fixed = set(universe_lits).difference(trial_cxp)
    #
    #         key = frozenset(trial_fixed)
    #         status = cache.get(key)
    #         if status is None:
    #             status = self._query_status_tabular(trial_fixed, epsilon)
    #             cache[key] = status
    #
    #         # still insufficient => lit not needed in CXP
    #         if status == "insufficient":
    #             cxp.remove(lit)
    #         elif status == "timeout":
    #             return None  # unknown
    #     return cxp
    #
    # def enumerate_all_axps_tabular(self, epsilon, stop_on_timeout=True):
    #     """
    #     Enumerate all subset-minimal AXPs for the current tabular instance.
    #     Returns: (axps_as_feature_indices, timed_out)
    #     """
    #     assert self.datatype == "tabular"
    #
    #     n = self.instance.shape[0]
    #     universe = [i + 1 for i in range(n)]  # Hitman expects positive ints
    #     axps = []
    #     cxps = []
    #     cache = {}
    #     timed_out = [False]
    #     timer = None
    #
    #     def interrupt():
    #         timed_out[0] = True
    #
    #     if self.time_limit is not None:
    #         timer = Timer(self.time_limit, interrupt)
    #         timer.start()
    #     try:
    #         with Hitman(bootstrap_with=[universe], htype="lbx") as hitman:
    #             while True:
    #                 if timed_out[0]:
    #                     break
    #                 hset = hitman.get()
    #                 if hset is None:
    #                     break
    #
    #                 fixed = set(hset)
    #                 key = frozenset(fixed)
    #                 status = cache.get(key)
    #                 if status is None:
    #                     status = self._query_status_tabular(fixed, epsilon)
    #                     cache[key] = status
    #
    #                 if status == "timeout":
    #                     timed_out[0] = True
    #                     if stop_on_timeout:
    #                         break
    #                     # conservative: block current seed and continue
    #                     hitman.block(fixed)
    #                     continue
    #
    #                 if status == "sufficient":
    #                     axp = self._extract_axp_tabular(fixed, epsilon, cache)
    #                     if axp is None:
    #                         timed_out[0] = True
    #                         if stop_on_timeout:
    #                             break
    #                         hitman.block(fixed)
    #                         continue
    #                     axps.append(sorted([lit - 1 for lit in axp]))  # back to 0-based feature ids
    #                     hitman.block(axp)
    #                 else:
    #                     cxp = self._extract_cxp_tabular(fixed, universe, epsilon, cache)
    #                     if cxp is None:
    #                         timed_out[0] = True
    #                         if stop_on_timeout:
    #                             break
    #                         hitman.block(fixed)
    #                         continue
    #                     cxps.append(sorted([lit - 1 for lit in cxp]))
    #                     hitman.hit(cxp)
    #
    #     finally:
    #         if self.time_limit is not None:
    #             if timer is not None:
    #                 timer.cancel()
    #         print("exiting enumeration")
    #
    #
    #     return axps,cxps, timed_out
    #
    def axps_to_fox_abd_strings(self, axps, cxps, class_value_map=None):
        mapped_label = self._format_class_value(self.label, class_value_map=class_value_map)
        abd = []
        for axp in axps:
            lhs = [f"{self.feature_names[i]} = {self.instance[i]}" for i in axp]
            if not lhs:
                abd.append(f"IF TRUE THEN label = {mapped_label}")
            else:
                abd.append(f"IF {' AND '.join(lhs)} THEN label = {mapped_label}")
        con = []
        for cxp in cxps:
            feats = cxp["features"]
            vals = cxp["values"]
            lhs = [f"{self.feature_names[i]} = {round(vals[i], 5)}" for i in feats]
            target_class = cxp.get("target_class")
            if target_class is not None:
                mapped_target = self._format_class_value(target_class, class_value_map=class_value_map)
                con.append(f"IF {' AND '.join(lhs)} THEN label = {mapped_target}")
            else:
                con.append(f"IF {' AND '.join(lhs)} THEN label != {mapped_label}")
        return {"abd": abd, "con": con}

    def _build_explained_instance_string(self, class_value_map=None):
        mapped_label = self._format_class_value(self.label, class_value_map=class_value_map)
        if self.datatype == "tabular" and self.feature_names is not None:
            lhs = [f"{self.feature_names[i]} = {round(float(self.instance[i]), 5)}" for i in range(len(self.instance))]
        else:
            flat_instance = np.asarray(self.instance).reshape(-1)
            lhs = [f"feature_{i} = {round(float(v), 5)}" for i, v in enumerate(flat_instance)]
        return f"IF {' AND '.join(lhs)} THEN label = {mapped_label}"

    def ffa_like_fox(self, explanation_list):
        """
        Same logic as FoX.ffa but local to VeriX.
        """
        print(f"DEBUG: ffa_like_fox received {len(explanation_list['abd'])} abductive explanations")
        axps = map(
            lambda s: s.split("IF ", maxsplit=1)[-1].rsplit(" THEN ", maxsplit=1)[0].split(" AND "),
            explanation_list["abd"]
        )
        lit_count = defaultdict(float)
        axps_ = [
            [t.split(" = ", maxsplit=1)[0].strip() for t in xp if t.strip() and t.strip() != "TRUE"]
            for xp in axps
        ]
        print(f"DEBUG: axps_ has {len(axps_)} AXPs after parsing")
        if not axps_:
            return {}
        nof_axps = len(axps_)
        for axp in axps_:
            if not axp:
                continue
            weight = 1 / len(axp)
            for lit in axp:
                lit_count[lit] += weight
        lit_count = {lit: cnt/nof_axps for lit, cnt in lit_count.items()}
        print(lit_count)
        return lit_count

    def save_ffa_graph(self, f2imprt):
        sorted_items = sorted(f2imprt.items(), key=lambda x: (abs(x[1]), x[0]))

        names_all = [k for k, v in sorted_items]
        values_all = [v for k, v in sorted_items]

        top5_items = sorted(f2imprt.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top5_items = sorted(top5_items, key=lambda x: (abs(x[1]), x[0]))

        names_top5 = [k for k, v in top5_items]
        values_top5 = [v for k, v in top5_items]

        def save_feature_plot(names, values, filename):
            plt.rcParams['axes.linewidth'] = 2

            fig, ax = plt.subplots()
            fig.set_size_inches(4, 4)

            for n, v in zip(names, values):
                if v > 0:
                    ax.barh(y=[n], width=[v], alpha=0.4, height=0.3,
                            color=(0.2, 0.4, 0.6, 0.6))
                else:
                    ax.barh(y=[n], width=[v], alpha=0.8, height=0.3,
                            color='orange')

            # Despine
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_visible(False)

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.tick_params(axis='y', pad=3, labelsize=15)

            for h, (n, v) in enumerate(zip(names, values)):
                ax.text(v, h + .18, f'{v:.2f}', color='black',
                        horizontalalignment='left' if v > 0 else 'right',
                        fontsize=10)

                ax.text(-.003 if v > 0 else .003, h - .05, n, color='black',
                        horizontalalignment='right' if v > 0 else 'left',
                        fontsize=10)

            plt.savefig(filename, bbox_inches='tight')
            plt.close()

        # Save both figures
        all_path = './ffa_all.png'
        top5_path = './ffa_top5.png'
        save_feature_plot(names_all, values_all, all_path)
        save_feature_plot(names_top5, values_top5, top5_path)

        with open(all_path, "rb") as f:
            self.ffa_all_img_src = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
        with open(top5_path, "rb") as f:
            self.ffa_top5_img_src = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

# added by claude
    def show_in_jupyter(self, show_both_exp=False) -> None:
        self.accordion.set_title(index=0, title=f"Instance")
        if self.stakeholder_name == 'model_subject':
            synthesis_html = self.synthesis_exp_html or (
                '<div style="padding:12px; border:1px solid #e0a800; border-radius:6px; '
                'background:#fff8e1; color:#7a5c00;">'
                '<b>⚠ Synthesised explanation unavailable</b><br>'
                'No synthesised explanation could be generated.'
                '</div>'
            )
            self.tab_nest.children = [widgets.HTML(value=synthesis_html)]
            self.tab_nest.set_title(0, "Synthesised Exp.")
            from IPython.display import display
            display(self.accordion)
            return
        abd_exp_html = widgets.HTML(value=self.abd_exp_html)
        con_exp_html = widgets.HTML(value=self.con_exp_html)
        children = [abd_exp_html, con_exp_html]
        titles = ["Abductive Exp.", "Contrastive Exp."]
        if self.synthesis_exp_html:
            children.append(widgets.HTML(value=self.synthesis_exp_html))
            titles.append("Synthesised Exp.")
        if self.ffa_fig is not None:
            children.append(self.ffa_fig)
            titles.append("Formal Feature Attribution")
        self.tab_nest.children = children
        for idx, title in enumerate(titles):
            self.tab_nest.set_title(idx, title)
        from IPython.display import display
        display(self.accordion)

    def exp_to_html(self, exp_list=None, exp_type=None):
        for exp_str in exp_list:
            exp = self.exp_mapping(exp_str)
            if self.explained_instance == "":
                self.explained_instance = HtmlString(
                    list_of_pair=self.get_explained_instance_pairs(),
                    exp_type='abd',
                    is_explained_instance=True,
                    prediction_label=self.prediction_label
                ).get_html()
            self.instance_info_html += self.explained_instance
            if exp_type == "abd":
                self.abd_exp_html += HtmlString(list_of_pair=exp, exp_type="abd",prediction_label=self.prediction_label).get_html()
            elif exp_type == "con":
                self.con_exp_html += HtmlString(list_of_pair=exp, exp_type="con",prediction_label=self.prediction_label).get_html()

    def _format_class_value(self, class_value, class_value_map=None):
        mapping = class_value_map if class_value_map is not None else self.class_value_map
        if not mapping:
            return str(class_value)
        candidates = [class_value, str(class_value)]
        try:
            fv = float(class_value)
            candidates.extend([fv, str(fv)])
            if fv.is_integer():
                iv = int(fv)
                candidates.extend([iv, str(iv)])
        except (TypeError, ValueError):
            pass
        for cand in candidates:
            if cand in mapping:
                return str(mapping[cand])
        return str(class_value)

    def exp_mapping(self, if_else_text):
        mapped = []
        feature_value = if_else_text.split('THEN')[0]
        feature_value = feature_value.split('AND')
        feature_value = [word.strip("IF ") for word in feature_value]
        for fea_val_pair in feature_value:
            fea_val_pair = fea_val_pair.strip()
            if not fea_val_pair:  # Skip empty strings
                continue
            if '!=' in fea_val_pair:
                fea_val = fea_val_pair.split('!=', 1)
                if len(fea_val) == 2:
                    val_str = fea_val[1].strip()
                    val = 1.0 if val_str.lower() == 'true' else (0.0 if val_str.lower() == 'false' else round(float(val_str), 5))
                    mapped.append([fea_val[0].strip(), f"!= {val}"])
            else:
                fea_val = fea_val_pair.split('=', 1)
                if len(fea_val) == 2:
                    val_str = fea_val[1].strip()
                    val = 1.0 if val_str.lower() == 'true' else (0.0 if val_str.lower() == 'false' else round(float(val_str), 5))
                    mapped.append([fea_val[0].strip(), val])
        label_value = if_else_text.split('THEN')[1].strip().split("=", 1)
        if len(label_value) == 2:
            mapped.append([label_value[0].strip(), self._format_class_value(label_value[1].strip())])
        return mapped

    def get_explained_instance_pairs(self):
        """Build the explained instance pairs in the same format exp_mapping returns."""
        pairs = []
        for i, val in enumerate(self.instance):
            pairs.append([self.feature_names[i], round(float(val), 5)])
        pairs.append(["label", self._format_class_value(self.label)])
        return pairs

    def _get_single_shot_axp(self, epsilon):
        """Compute single-shot AXP for tabular data when not already computed."""
        sat_set = []
        unsat_set = []
        timeout_set = []
        instance = self.instance

        for feat in self.inputVars:
            for i in self.inputVars:
                if i == feat or i in unsat_set:
                    self.mara_model.setLowerBound(i, instance[i] - epsilon if epsilon else -1e9)
                    self.mara_model.setUpperBound(i, instance[i] + epsilon if epsilon else 1e9)
                else:
                    self.mara_model.setLowerBound(i, instance[i])
                    self.mara_model.setUpperBound(i, instance[i])

            for j in range(len(self.outputVars)):
                if j != self.label:
                    self.mara_model.addInequality([self.outputVars[self.label], self.outputVars[j]],
                                                  [1, -1], -1e-6,
                                                  isProperty=True)
                    exit_code, vals, stats = self.mara_model.solve(options=self.options, verbose=False)
                    self.mara_model.additionalEquList.clear()
                    if exit_code == 'sat' or exit_code == 'TIMEOUT':
                        break
                    elif exit_code == 'unsat':
                        continue

            self.mara_model.clearProperty()

            if exit_code == 'unsat':
                unsat_set.append(feat)
            elif exit_code == 'TIMEOUT':
                timeout_set.append(feat)
            elif exit_code == 'sat':
                sat_set.append(feat)

        return sorted(sat_set)

    def enumerate_all_axps_tabular_fast(self, epsilon, skip_axp=None, max_seconds=None, num_axps=None):
        n = self.instance.shape[0]
        universe = [i + 1 for i in range(n)]
        print(f"n:{n}")
        axps, cxps = [], []

        known_sufficient = []
        known_insufficient = []
        cache = {}

        def fast_status(fixed):
            fset = set(fixed)

            # monotonic pruning
            for s in known_sufficient:
                if s.issubset(fset):
                    return "sufficient", None

            for s,w in known_insufficient:
                if fset.issubset(s):
                    extra = s - fset
                    base_values = dict(w.get("values", {})) if isinstance(w, dict) else {}
                    for lit in extra:
                        base_values[lit] = float(self.instance[lit - 1])
                    return "insufficient", {
                        "values": base_values,
                        "target_class": w.get("target_class") if isinstance(w, dict) else None
                    }

            key = frozenset(fset)
            if key in cache:
                return cache[key]

            status, witness = self._query_status_tabular(fset, epsilon)
            cache[key] = (status, witness)

            if status == "sufficient":
                known_sufficient.append(fset)
            elif status == "insufficient":
                known_insufficient.append((fset,witness))

            return status,witness

        timed_out = [False]
        timer = None

        def interrupt():
            timed_out[0] = True

        time_budget = self.time_limit if max_seconds is None else max_seconds
        if time_budget is not None:
            timer = Timer(time_budget, interrupt)
            timer.start()
        try:
            with Hitman(bootstrap_with=[universe], htype="lbx") as hitman:
                while True:
                    if timed_out[0]:
                        break
                    hset = hitman.get()
                    if hset is None:
                        break

                    fixed = set(hset)
                    status, witness = fast_status(fixed)

                    if status == "sufficient":
                        axp = set(fixed)

                        # randomized shrinking (faster)
                        for lit in list(axp):
                            trial = axp - {lit}
                            trial_status, _ = fast_status(trial)
                            if trial_status == "sufficient":
                                axp.remove(lit)

                        axps.append(sorted([l - 1 for l in axp]))
                        hitman.block(axp)
                        # If num_axps is set and we found a non-empty one, check if we have enough
                        if num_axps is not None and axp and len(axps) >= num_axps:
                            break


                    elif status == "insufficient":
                        cxp = set(universe) - fixed
                        cxp_payload = {
                            "values": dict(witness.get("values", {})),
                            "target_class": witness.get("target_class")
                        }

                        for lit in list(cxp):
                            trial_cxp = cxp - {lit}
                            trial_fixed = set(universe) - trial_cxp
                            trial_status, trial_witness = fast_status(trial_fixed)

                            if trial_status == "insufficient":
                                cxp.remove(lit)
                                cxp_payload = {
                                    "values": dict(trial_witness.get("values", {})),
                                    "target_class": trial_witness.get("target_class")
                                }

                        cxps.append({
                            "features": sorted([l - 1 for l in cxp]),
                            "values": {l - 1: cxp_payload["values"][l] for l in cxp if l in cxp_payload["values"]},
                            "target_class": cxp_payload["target_class"],
                        })
                        hitman.hit(cxp)
                        # For stop_when_found: don't break on CXP, continue searching for AXPs

                    else:  # timeout
                        hitman.block(fixed)
        finally:
            if self.time_limit is not None:
                if timer is not None:
                    timer.cancel()
            print("exiting enumeration")

        return axps, cxps

def save_figure(image, path, cmap=None):
    """
    To plot figures.
    :param image: the image array of shape (width, height, channel)
    :param path: figure name.
    :param cmap: 'gray' if to plot gray scale image.
    :return: an image saved to the designated path.
    """
    fig = plt.figure()
    ax = plt.Axes(fig, [-0.5, -0.5, 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if cmap is None:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=cmap)
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)