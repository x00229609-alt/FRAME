from __future__ import print_function

from .lrxp import LRExplainer
from .options import Options
from .rndmforest import XRF, ModelDataset
from .html_string import HtmlString

import importlib.util
import numpy as np
import pandas as pd
import ipywidgets as widgets
import collections
import matplotlib.pyplot as plt
import base64
from html import escape
from pathlib import Path


def _load_gemini_synthesiser():
    module_path = Path(__file__).resolve().parents[1] / "XReason" / "gemini_synthesiser.py"
    spec = importlib.util.spec_from_file_location("xreason_gemini_synthesiser", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.synthesise_explanation

class FoX(object):
    """Formal explainer for sklearn Random Forest and Logistic Regression models.

    Parameters
    ----------
    model_path : str
        Path to a ``.pkl`` file containing the trained sklearn RF or LR model.
    global_model_name : str
        ``'RF'`` for Random Forest or ``'LR'`` for Logistic Regression.
    xtype : str
        ``'abd'`` (abductive) or ``'con'`` (contrastive). Default ``'abd'``.
    feature_names : list of str, optional
        Feature names. Inferred from ``model.feature_names_in_`` when available.
    class_names : list, optional
        Class label strings. Inferred from ``model.classes_`` when available.
    label_name : str
        Name of the target variable shown in explanations. Default ``'class'``.
    feature_bounds : dict or list of (min, max), optional
        Per-feature value bounds used by LR contrastive explanations.
        Dict: ``{feature_index: (min_val, max_val)}``.
        List: ``[(min_val, max_val), ...]`` in feature order.
        When omitted, defaults to ``[-1e6, 1e6]`` per feature.
    time_limit : float, optional
        Solver time limit in seconds.
    prediction_label : str
        Label shown in HTML output for the prediction. Default ``'Prediction'``.
    """

    def __init__(self, model_path, global_model_name="LR",
                 xtype='abd',
                 feature_names=None, class_names=None, label_name='class',
                 feature_bounds=None,
                 time_limit=None, prediction_label="Prediction", stakeholder_name='developer',
                 class_value_map=None):
        import pickle
        with open(model_path, 'rb') as f:
            _model = pickle.load(f)
        self.global_model_name = global_model_name
        self.stakeholder_type = stakeholder_name
        self.feature_bounds = feature_bounds
        self.options = Options(
            global_model_name=global_model_name,
            xtype=xtype,
            xnum='all',
            time_limit=time_limit,
            classifier=model_path,
        )
        self.dataset = ModelDataset(
            model=_model,
            feature_names=feature_names,
            class_names=class_names,
            label_name=label_name,
        )
        self.explainer = None
        self.tab_nest = widgets.Tab()
        self.accordion = widgets.Accordion(children=[self.tab_nest])
        self.explained_instance = ""
        self.abd_con_exp_html = ""
        self.abd_exp_html = ""
        self.con_exp_html = ""
        self.instance_info_html = ""
        self.ffa_fig = None
        self.ffa_all_img_src = ""
        self.ffa_top5_img_src = ""
        self.synthesis_exp_html = ""
        self.pred = None
        self.prediction_label = prediction_label
        self.class_value_map = class_value_map if class_value_map is not None else {}

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

    def exp_to_html(self, exp_list=None, exp_type=None, explained_instance=None, class_value_map=None):
        for exp in exp_list:
            exp = self.exp_mapping(exp, class_value_map=class_value_map)
            if self.explained_instance == "":
                self.explained_instance = HtmlString(list_of_pair=explained_instance, exp_type=self.options.xtype, is_explained_instance=True, prediction_label=self.prediction_label).get_html()
            self.instance_info_html += self.explained_instance
            if exp_type == "abd":
                self.abd_exp_html += HtmlString(list_of_pair=exp, exp_type="abd", prediction_label=self.prediction_label).get_html()
            elif exp_type == "con":
                self.con_exp_html += HtmlString(list_of_pair=exp, exp_type="con", prediction_label=self.prediction_label).get_html()

    def show_in_jupyter(self, show_both_exp=False) -> None:
        accordion_title = f"Explained Instance"
        if self.stakeholder_type == 'model_subject':
            self.accordion.set_title(index=0, title=accordion_title)
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
            return display(self.accordion)
        if show_both_exp:
            self.accordion.set_title(index=0, title=accordion_title)
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
            for index, title in enumerate(titles):
                self.tab_nest.set_title(index, title)
        else:
            self.accordion.set_title(index=0, title=accordion_title)
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
            for index, title in enumerate(titles):
                self.tab_nest.set_title(index, title)
        from IPython.display import display
        return display(self.accordion)

    def explain(self, sample, in_jupyter=False, prediction_label=None, class_value_map=None):
        """Explain a prediction for the given sample.

        Parameters
        ----------
        sample : array-like or pandas Series
            Instance to explain (feature values only, no label).
        in_jupyter : bool
            Render output as an interactive Jupyter widget.
        prediction_label : str, optional
            Override self.prediction_label for this call only.
        """
        if in_jupyter:
            self.options.in_jupyter = True
        options = self.options

        if options.xtype:
            print('\nExplaining the {0} model...\n'.format(
                'logistic regression' if options.global_model_name == 'LR' else 'random forest'))

            # Convert sample to Series aligned to dataset feature names
            if isinstance(sample, pd.Series):
                inst = sample
            else:
                arr = np.asarray(sample).ravel()
                inst = pd.Series(arr, index=self.dataset.feature_names)

            if options.global_model_name == 'RF':
                self.explainer = XRF(self.dataset, options)
            elif options.global_model_name == 'LR':
                self.explainer = LRExplainer(self.dataset, options,
                                             feature_bounds=self.feature_bounds)

            _, _, explained_instance, explanation_list, explanation_size_list, self.pred = \
                self.explainer.explain(inst)
            self._render_explanation(in_jupyter, explained_instance, explanation_list, explanation_size_list,
                                     prediction_label=prediction_label,
                                     class_value_map=class_value_map)

    def _render_explanation(self, in_jupyter, explained_instance, explanation_list, explanation_size_list,
                            prediction_label=None, class_value_map=None):
        """Shared rendering logic for both direct-sample and legacy CSV paths."""
        use_prediction_label = prediction_label if prediction_label is not None else self.prediction_label
        use_class_value_map = class_value_map if class_value_map is not None else self.class_value_map

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

        def get_appropriate_n_expl(axps, cxps, stakeholder):
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

        # axps_sorted = sorted(explanation_list['abd'], key=len)
        # cxps_sorted = sorted(explanation_list['con'], key=_exp_len)
        # axps_selected = axps_sorted[:5]
        # cxps_selected = cxps_sorted[:5]
        axps = explanation_list['abd']
        cxps = explanation_list['con']
        axps_selected, cxps_selected = get_appropriate_n_expl(axps, cxps, self.stakeholder_type)
        self.synthesis_exp_html = ""

        if in_jupyter:
            explained_instance_mapped = self.exp_mapping(explained_instance, class_value_map=use_class_value_map)
            if self.options.xnum not in (-1, 'all'):
                expl = explanation_list[self.options.xtype][0]
                explanation = self.exp_mapping(expl, class_value_map=use_class_value_map)
                if self.explained_instance == "":
                    self.explained_instance = HtmlString(list_of_pair=explained_instance_mapped,
                                                         exp_type=self.options.xtype,
                                                         is_explained_instance=True,
                                                         prediction_label=use_prediction_label
                                                         ).get_html()
                self.instance_info_html += self.explained_instance
                self.abd_con_exp_html += HtmlString(list_of_pair=explanation,
                                                    exp_type=self.options.xtype,
                                                    prediction_label=use_prediction_label
                                                    ).get_html()
                if self.stakeholder_type == 'model_subject':
                    try:
                        synthesise_explanation = _load_gemini_synthesiser()
                        print("[FRAME DEBUG][model_subject] AXP for synthesis:", axps_selected[0])
                        print("[FRAME DEBUG][model_subject] CXPs for synthesis:", cxps_selected[:2])
                        synthesised_response = synthesise_explanation(
                            axp_string=axps_selected[0],
                            cxp_strings=cxps_selected[:2],
                            instance_explanation_string=explained_instance,
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
                self.show_in_jupyter()
            else:
                # enumeration
                _saved_label = self.prediction_label
                self.prediction_label = use_prediction_label
                self.exp_to_html(exp_list=axps_selected, exp_type='abd',
                                 explained_instance=explained_instance_mapped,
                                 class_value_map=use_class_value_map)
                self.exp_to_html(exp_list=cxps_selected, exp_type='con',
                                 explained_instance=explained_instance_mapped,
                                 class_value_map=use_class_value_map)
                self.prediction_label = _saved_label

                if self.stakeholder_type == 'model_subject' and axps_selected and cxps_selected:
                    try:
                        synthesise_explanation = _load_gemini_synthesiser()
                        print("[FRAME DEBUG][model_subject] AXP for synthesis:", axps_selected[0])
                        print("[FRAME DEBUG][model_subject] CXPs for synthesis:", cxps_selected[:2])
                        synthesised_response = synthesise_explanation(
                            axp_string=axps_selected[0],
                            cxp_strings=cxps_selected[:2],
                            instance_explanation_string=explained_instance,
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

                self.ffa_fig = None
                if self.stakeholder_type == 'developer':
                    ffa = self.ffa(explanation_list)
                    if ffa != {}:
                        self.save_ffa_graph(ffa)
                        exp_type_full = "Formal Feature Attribution"
                        color = "rgba(237,34,14,255)" if str(self.pred) == "True" else "rgba(96,217,55,255)"
                        equal_sign = "&nbsp;&nbsp;&nbsp;&nbsp;="
                        label_title = use_prediction_label
                        pred_display = self._format_class_value(self.pred, class_value_map=use_class_value_map)
                        ffa_html = f'''
                                    <div class="box">
                                        <div class="inner-box">
                                            <text class="title"> {exp_type_full}</text>
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
                                        .img2 {{ display: none; }}
                                        #toggle:checked ~ .img1 {{ display: none; }}
                                        #toggle:checked ~ .img2 {{ display: inline; }}
                                        .show-top5 {{ display: none; }}
                                        #toggle:checked + .toggle-label .show-all {{ display: none; }}
                                        #toggle:checked + .toggle-label .show-top5 {{ display: inline; }}
                                        .toggle-label {{
                                            display: inline-block; padding: 6px 12px;
                                            border: 1px solid #888; border-radius: 4px;
                                            background: #f5f5f5; cursor: pointer; user-select: none;
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
                    else:
                        self.ffa_fig = widgets.HTML(value=(
                            '<div style="padding:12px; border:1px solid #e0a800; border-radius:6px; '
                            'background:#fff8e1; color:#7a5c00;">'
                            '<b>⚠ Formal Feature Attribution unavailable</b><br>'
                            'No explanations were found within the time limit. '
                            'Try increasing <code>time_limit</code> and re-running.'
                            '</div>'
                        ))
                self.show_in_jupyter(show_both_exp=True)
        else:
            if self.options.xnum not in (-1, 'all'):
                exp_type_name = "Abductive" if self.options.xtype == "abd" else "Contrastive"
                expl = explanation_list[self.options.xtype][0]
                explanation_size = explanation_size_list[self.options.xtype][0]
                print("Explained Instance\n", explained_instance, f"\n\n{exp_type_name} Explanation\n", expl, "\n\n", explanation_size, "\n")
            else:
                print("Explained Instance\n ", explained_instance)
                for xtype in ['abd', 'con']:
                    exp_type_name = "Abductive" if xtype == "abd" else "Contrastive"
                    print(f"\n{exp_type_name} Explanation")
                    for i, expl in enumerate(explanation_list[xtype]):
                        explanation_size = explanation_size_list[xtype][i]
                        print(' ', expl, "\n\n ", explanation_size, "\n\n")
                if self.stakeholder_type == 'developer':
                    ffa = self.ffa(explanation_list)
                    if ffa:
                        print('FFA:\n{}'.format(ffa))
                    else:
                        print("⚠ Formal Feature Attribution unavailable: no explanations were found "
                              "within the time limit. Try increasing time_limit and re-running.")

    def exp_mapping(self, if_else_text, class_value_map=None):
        # use list to preserve the order of the if-else statements
        mapped = []
        # map features
        feature_value = if_else_text.split('THEN')[0]
        feature_value = feature_value.split('AND')
        feature_value = [word.strip("IF ") for word in feature_value]
        for fea_val_pair in feature_value:
            fea_val = fea_val_pair.split('=')
            val_str = fea_val[1].strip()
            if val_str.lower() == 'true':
                val = 1.0
            elif val_str.lower() == 'false':
                val = 0.0
            else:
                val = round(float(val_str), 5)
            mapped.append([fea_val[0].strip(), val])
        # map label
        label_value = if_else_text.split('THEN')[1].strip().split("=")
        mapped.append([label_value[0].strip(), self._format_class_value(label_value[1].strip(), class_value_map=class_value_map)])
        return mapped
    
    def ffa(self, explanation_list):
        """
        unweighted feature attribution
        """
        print("Entering Feature Attribution")
        if not explanation_list.get('abd'):
            print("No abductive explanations found — FFA cannot be computed. "
                  "Try increasing the time_limit.")
            return {}

        axps = map(lambda l: l.split('IF ', maxsplit=1)[-1].rsplit(' THEN ', maxsplit=1)[0].split(' AND '), 
                   explanation_list['abd'])
        print(len(explanation_list['abd']))

        axps_ = []
        for xp in axps:
            filtered = [t.split(' = ', maxsplit=1)[0].strip() for t in xp if t.strip() and t.strip() != 'TRUE']
            if filtered:
                axps_.append(filtered)

        if not axps_:
            print("All explanations are trivially true — no features to attribute. "
                  "Try increasing the time_limit to find more specific explanations.")
            return {}

        sizes = [len(axp) for axp in axps_]
        print(f"min lits: {min(sizes)}, max lits: {max(sizes)}, avg: {sum(sizes) / len(sizes):.2f}")

        lit_count = collections.defaultdict(lambda: 0)
        nof_axps = len(axps_)
        for axp in axps_:
            weight = 1 / len(axp)
            for lit in axp:
                lit_count[lit] += weight
        lit_count = {lit: cnt/nof_axps for lit, cnt in lit_count.items()}
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

        # Save both figures and embed as data URLs for stable Jupyter rendering.
        all_path = './ffa_all.png'
        top5_path = './ffa_top5.png'
        save_feature_plot(names_all, values_all, all_path)
        save_feature_plot(names_top5, values_top5, top5_path)

        with open(all_path, "rb") as f:
            self.ffa_all_img_src = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
        with open(top5_path, "rb") as f:
            self.ffa_top5_img_src = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
