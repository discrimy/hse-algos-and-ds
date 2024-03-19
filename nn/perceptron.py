from collections import Counter
from pathlib import Path

import numpy as np
from rich.progress import track
from scipy.special import expit


class Perceptron:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, learning_rate: float):
        self.learning_rate = learning_rate
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.input_nodes = input_nodes

        self.weights_ih = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
        # Sigmoid only, generalize later
        self.activation_func = expit

    def train(self, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        # Straight run
        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)
        output_inputs = np.dot(self.weights_ho, hidden_outputs)
        output_outputs = self.activation_func(output_inputs)
        # Layers error calculation
        output_errors = targets - output_outputs
        hidden_errors = np.dot(self.weights_ho.T, output_errors)
        # Weights correction
        self.weights_ho += self.learning_rate * np.dot(
            # Sigmoid only, generalize later
            output_errors * output_outputs * (1 - output_outputs),
            hidden_outputs.T
        )
        self.weights_ih += self.learning_rate * np.dot(
            # Sigmoid only, generalize later
            hidden_errors * hidden_outputs * (1 - hidden_outputs),
            inputs.T
        )

        return output_outputs

    def query(self, inputs: np.ndarray) -> np.ndarray:
        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)
        output_inputs = np.dot(self.weights_ho, hidden_outputs)
        output_outputs = self.activation_func(output_inputs)

        return output_outputs


def main() -> None:
    input_nodes = 28 * 28
    hidden_nodes = 200
    output_nodes = 10

    learning_rate = 0.3
    nn = Perceptron(input_nodes, hidden_nodes, output_nodes, learning_rate)

    training_data_list = Path('mnist_train.csv').read_text().splitlines()
    for _ in range(1):
        for record in track(training_data_list, 'Train'):
            answer, *image_str = record.split(',')
            inputs_list = (np.asfarray(image_str) / 255 * 0.99) + 0.01
            targets_list = np.zeros(output_nodes) + 0.01
            targets_list[int(answer)] = 0.99

            inputs = np.array(inputs_list, ndmin=2).T
            targets = np.array(targets_list, ndmin=2).T
            outputs = nn.train(inputs, targets)
            error = 0.5 * ((targets - outputs)**2).sum()
            # print(f'{error:.6f}')
    test_data_list = Path('mnist_test.csv').read_text().splitlines()
    answers = []
    for vals in test_data_list:
        answer, *image_raw = vals.split(',')
        answer = int(answer)
        inputs_list = (np.asfarray(image_raw) / 255 * 0.99) + 0.01
        inputs = np.array(inputs_list, ndmin=2).T
        outputs = nn.query(inputs)
        nn_answer = outputs.argmax()
        if nn_answer == answer:
            answers.append(1)
        else:
            answers.append(0)
    counter = Counter(answers)
    print(counter[1] / (counter[0] + counter[1]))


if __name__ == '__main__':
    main()
