from XReason import *
from foxplainer import *
from foxplainer.explainer import FoX
from VeriX import VeriX

class FRAME(object):
    """
        The main wrapper class to encode/explain different model types.
    """
    def __init__(self, model_type, model_path, categorical_values_dict=None, stakeholder_name="decision_maker", datatype="tabular",
                 feature_bounds=None, feature_names=None, time_limit=60, one_hot_groups=None,
                 class_value_map=None):


        """
         To initialize the FRAME class.

         :param instance: an instance of the data, tabular array or image array of shape (width, height, channel).
         :param model_path: the path to the model.
         :param stakeholder_name: 'developer' or 'decision_maker' or 'model_subject' , determines the explanation type
         :param categorical_values_dict: a dictionary to store the categorical columns and their potential values.
         :param datatype: 'tabular' or 'greyscale' or 'rgb' (greyscale or rgb only supported for VeriX)
        :param feature_bounds: a dictionary to store optional feature bounds for each feature.
        :param feature_names: a dictionary to store the feature names for each feature. only used for logistic regression.
         """

        if model_type not in ['xgboost','logistic_regression','random_forest','neural_network']:
            raise ValueError("model_type must be one of 'xgboost', 'logistic_regression', 'random_forest', or 'neural_network'")
        self.model_type = model_type
        self.model_path = model_path
        self.categorical_values_dict = categorical_values_dict
        self.time_limit = time_limit
        self.datatype = datatype
        self.feature_bounds = feature_bounds
        self.feature_names = feature_names
        self.one_hot_groups = one_hot_groups
        self.class_value_map = class_value_map
        if stakeholder_name.lower() not in ['developer', 'decision_maker', 'model_subject']:
            raise ValueError("stakeholder_type must be one of 'developer', 'decision_maker', or 'model_subject'")
        self.stakeholder_name = stakeholder_name.lower()
        # now the specific setups
        if self.model_type == 'xgboost':
            encoded_model = XGBooster(
                from_model=self.model_path,
                categorical_values_dict=self.categorical_values_dict,
                feature_bounds=self.feature_bounds,
                stakeholder_name= self.stakeholder_name,
                class_value_map=self.class_value_map
            )
            encoded_model.encode()
            self.encoded_model = encoded_model
        elif model_type == 'logistic_regression':
            if(feature_bounds is None or feature_names is None):
                raise ValueError("feature_bounds and feature_names cannot be None for logistic regression models.")
            encoded_model = FoX(global_model_name="LR",
                model_path=self.model_path,
                time_limit = self.time_limit,
                feature_bounds=self.feature_bounds,
                feature_names=list(self.feature_names),
                stakeholder_name=self.stakeholder_name,
                class_value_map=self.class_value_map)
            self.encoded_model = encoded_model
        elif model_type == 'random_forest':
            encoded_model = FoX(global_model_name="RF",
                model_path=self.model_path,
                time_limit = self.time_limit,
                feature_bounds=self.feature_bounds,
                feature_names=list(self.feature_names),
                stakeholder_name=self.stakeholder_name,
                class_value_map=self.class_value_map)
            self.encoded_model = encoded_model
        elif model_type == 'neural_network':
            if(self.datatype != 'tabular'):
                plot_original_bool = True
            else:
                plot_original_bool = False
            encoded_model = VeriX(
                datatype=self.datatype,
                feature_names=self.feature_names,
                model_path=self.model_path,
                plot_original=plot_original_bool,
                categorical_values_dict=self.categorical_values_dict,
                one_hot_groups=self.one_hot_groups,
                time_limit=self.time_limit,
                in_jupyter = True,
                stakeholder_name=self.stakeholder_name,
                class_value_map=self.class_value_map
            )
            self.encoded_model = encoded_model


    def explain(self,instance,prediction_label='prediction', epsilon=None):
        if self.model_type == 'neural_network':
            if epsilon is None:
                raise ValueError("epsilon must be provided when explaining neural_network models.")
            return self.encoded_model.explain(
                instance=instance,
                epsilon=epsilon,
                in_jupyter=True,
                prediction_label=prediction_label,
                class_value_map=self.class_value_map
            )
        return self.encoded_model.explain(
            sample=instance,
            in_jupyter=True,
            prediction_label=prediction_label,
            class_value_map=self.class_value_map
        )
