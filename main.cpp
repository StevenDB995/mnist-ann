#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <algorithm>
#include "ANN.h"

using namespace std;

int main() {
	vector< vector<float> > X_train;
	vector< vector<float> > Y_train;

	ifstream trainFile("train.txt");

	if (trainFile.is_open())
	{
		cout << "Loading train data ...\n";
		string line;
		while (getline(trainFile, line))
		{
			int temp;
			vector<float> x;
			vector<float> y(10);

			stringstream ss(line);
			ss >> temp;
			y[temp] = 1;
			Y_train.push_back(y);

			for (int i = 0; i < 28 * 28; ++i) {
				ss >> temp;
				x.push_back(temp / 255.0);
			}
			X_train.push_back(x);
		}

		trainFile.close();
		cout << "Loading train data finished.\n";
	} 
	else 
		cout << "Unable to open file" << '\n';

	vector< vector<float> > X_test;
	vector<int> Y_test;

	ifstream testFile("test.txt");

	if (testFile.is_open())
	{
		cout << "Loading test data ...\n";
		string line;
		while (getline(testFile, line))
		{
			int temp;
			vector<float> x;
			int y;

			stringstream ss(line);
			ss >> y;
			Y_test.push_back(y);

			for (int i = 0; i < 28 * 28; ++i) {
				ss >> temp;
				x.push_back(temp / 255.0);
			}
			X_test.push_back(x);
		}

		testFile.close();
		cout << "Loading test data finished.\n";
	} 
	else 
		cout << "Unable to open file" << '\n';

	cout << "------------------------------\n";

	char mode;
	cout << "Choose user mode or tester mode\n";
	cout << "(u or t): ";
	cin >> mode;

	cout << "------------------------------\n";

	ANN ann(3);

	if (mode == 't')
	{	/* Tester mode */
		ann.setNeurons(0, 28 * 28);
		ann.setNeurons(1, 400);
		ann.setNeurons(2, 10);	
		ann.build();

		ann.setLearningRate(0.2);
		ann.setMinibatchSize(400);
		ann.setEpochs(50);
		ann.set_num_threads(4);

		ann.loadWeights("trained_weights_0.txt");
	}
	else
	{	/* User mode */
		int layers;
		int neurons;
		float r;
		int size;
		int epochs;
		int threads;

		cout << "Set the number of layers: ";
		cin >> layers;
		ann.setLayers(layers);

		cout << "Set the number of neurons at\n";
		for (int i = 0; i < layers; ++i) {
			cout << "Layer #" << (i + 1) << ": ";
			cin >> neurons;
			ann.setNeurons(i, neurons);
		}

		ann.build();

		cout << "Set learning rate: ";
		cin >> r;
		ann.setLearningRate(r);

		cout << "Set mini-batch size: ";
		cin >> size;
		ann.setMinibatchSize(size);

		cout << "Set the number of epochs to run: ";
		cin >> epochs;
		ann.setEpochs(epochs);

		cout << "Set the number of threads for parallelism: ";
		cin >> threads;
		ann.set_num_threads(threads);

		cout << "------------------------------\n";	

		ann.train(X_train, Y_train);
	}

	cout << "------------------------------\n";

	int correct = 0;
	cout << "Label\tPredict\tCorrect\n";

	for (int i = 0; i < X_test.size(); ++i) {
		vector<float> p = ann.predict(X_test[i]);
		int pDigit = max_element(p.begin(), p.end()) - p.begin();
		if (pDigit == Y_test[i])
			++correct;

		cout << Y_test[i] << '\t' << pDigit << '\t'
			<< boolalpha << (pDigit == Y_test[i]) << endl;
	}

	cout << "------------------------------\n";

	cout << "Prediction accuracy: "
		<< (float) correct / (float) X_test.size() << endl;	

	char store;
	cout << "Do you need to store the trained weights to local file system?\n";
	cout << "(y or n): ";
	cin >> store;
	if (store == 'y') {
		string postfix;
		cout << "Set a postfix for the output filename: ";
		cin >> postfix;
		ann.storeWeights("trained_weights_" + postfix + ".txt");
	}

	return 0;
}
