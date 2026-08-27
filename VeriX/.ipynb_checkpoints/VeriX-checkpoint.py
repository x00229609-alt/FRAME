import random
import numpy as np
import onnx
import onnxruntime as ort
from skimage.color import label2rgb
from matplotlib import pyplot as plt
from typing import Literal
import sys
sys.path.insert(0, "Marabou")
"""
After installing Marabou, load it from maraboupy.
"""
from maraboupy import Marabou
from maraboupy import MarabouNetworkONNX



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
                                    solveWithMILP=True)

    def __init__(self,
                 datatype,
                 instance,
                 model_path,
                 plot_original=True):
        """
        To initialize the VeriX class.
        :param datatype: 'greyscale' or 'rgb' or 'tabular'
        :param instance: an image array of shape (width, height, channel).
        :param model_path: the path to the neural network.
        :param plot_original: if True, then plot the original image.
        """
        self.datatype: Literal["greyscale", "rgb", "tabular"] = datatype
        self.instance = instance
        """
        Load the onnx model.
        """
        self.onnx_model = onnx.load(model_path)
        self.onnx_session = ort.InferenceSession(model_path)
        onnx_input_data = np.expand_dims(instance, axis=0) if instance.ndim == 1 else np.expand_dims(instance, axis=0)
        prediction = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: onnx_input_data})
        prediction = np.asarray(prediction[0])
        self.label = prediction.argmax()
        """
        Load the onnx model into Marabou.
        Note: to ensure sound and complete analysis, load the model before the softmax activation function;
        if the model is trained from logits directly, then load the whole model. 
        """
        self.mara_model = Marabou.read_onnx(model_path)
        if self.onnx_model.graph.node[-1].op_type == "Softmax":
            mara_model_output = self.onnx_model.graph.node[-1].input
        else:
            mara_model_output = None
        self.mara_model = Marabou.read_onnx(filename=model_path,
                                            outputNames=mara_model_output)
        if self.datatype in ["greyscale", "rgb"]:
            width, height = instance.shape[0], instance.shape[1]
            self.inputVars = np.arange(width * height)
        elif self.datatype == "tabular":
            self.inputVars = np.arange(instance.shape[0])

        self.outputVars = self.mara_model.outputVars[0].flatten()
        if plot_original and (self.datatype in ["greyscale","rgb"]):
            save_figure(image=instance,
                        path=f"original-predicted-as-{self.label}.png",
                        cmap="gray" if self.datatype == 'greyscale' else None)

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
                if plot_sensitivity and (self.datatype in ["greyscale","rgb"]):
                    save_figure(image=self.sensitivity, path=f'{self.datatype}-sensitivity-{self.traverse}.png')
            elif self.datatype == "tabular":
                random.seed(seed)
                random.shuffle(self.inputVars)
        elif self.traverse == "random" or self.datatype =="tabular":
            random.seed(seed)
            random.shuffle(self.inputVars)
        else:
            print("Traversal not supported: try 'heuristic' or 'random'.")

    def get_explanation(self,
                        epsilon,
                        plot_explanation=True,
                        plot_counterfactual=False,
                        plot_timeout=False):
        """
        To compute the explanation for the model and the neural network.
        :param epsilon: the perturbation magnitude.
        :param plot_explanation: if True, plot the explanation.
        :param plot_counterfactual: if True, plot the counterfactual(s).
        :param plot_timeout: if True, plot the timeout pixel(s).
        :return: an explanation, and possible counterfactual(s).
        """
        unsat_set = []
        sat_set = []
        timeout_set = []
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
                        if epsilon is None:
                            INF = 1e9
                            self.mara_model.setLowerBound(i, -INF)
                            self.mara_model.setUpperBound(i, INF)
                        else: #might need to make sure this makes sense later
                            self.mara_model.setLowerBound(i, instance[i]-epsilon)
                            self.mara_model.setUpperBound(i, instance[i]+epsilon)

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
                    else:
                        print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
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
            """
            self.mara_model.clearProperty()
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
                    else: # we can save this elsewhere later for the tooltip/hover feature if we have time
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