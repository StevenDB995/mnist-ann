#include "ANN.h"
#include <iostream>
#include <cmath>
#include <ctime>
#include <random>
#include <algorithm>
#include <fstream>
#include <omp.h>

using namespace std;

ANN::ANN(int layers)
{
	setLayers(layers);
}

void ANN::setLayers(int layers)
{
	this->layers = layers;
	neuron_num = new int[layers];

	a = new float *[layers];
	b = new float *[layers];
	err = new float *[layers];
	W = new float *[layers];
}

void ANN::setNeurons(int index, int neurons)
{
	neuron_num[index] = neurons;

	a[index] = new float[neurons];
	if (index > 0) {
		b[index] = new float[neurons];
		err[index] = new float[neurons];
	}
}

void ANN::build(void)
{
	random_device rd;
	mt19937 g(rd());

	for (int i = 1; i < layers; ++i) {
		int m = neuron_num[i];
		int n = neuron_num[i - 1];
		W[i] = new float[m * n];
	}

	for (int i = 1; i < layers; ++i) {
		int m = neuron_num[i];
		int n = neuron_num[i - 1];

		float range = 4.0 * sqrt(6.0) / sqrt(n + m);
		uniform_real_distribution<float> dis1(-range, range);
		normal_distribution<float> dis2(0, 1);

		for (int j = 0; j < m * n; ++j)
			W[i][j] = dis1(g);

		for (int j = 0; j < m; ++j)
			b[i][j] = dis2(g);
	}
}

void ANN::setLearningRate(float r)
{
	this->r = r;
}

void ANN::setMinibatchSize(int size)
{
	minibatchSize = size;
}

void ANN::setEpochs(int epochs)
{
	this->epochs = epochs;
}

void ANN::set_num_threads(int threads)
{
	thread_count = threads;
}

void ANN::feedForward(vector<float> &x)
{
	for (int j = 0; j < neuron_num[0]; ++j)
		a[0][j] = x[j];

	for (int i = 1; i < layers; ++i) {
		int m = neuron_num[i];
		int n = neuron_num[i - 1];

		#pragma omp parallel for num_threads(thread_count)
		for (int j = 0; j < m; ++j) {
			float z = 0;	
			for (int k = 0; k < n; ++k)
				z += W[i][j * n + k] * a[i - 1][k];
			z += b[i][j];
			a[i][j] = sigmoid(z);
		}
	}
}

void ANN::backpropagate(vector<float> &y)
{
	/* Find the errors of the last layer first */
	for (int j = 0; j < neuron_num[layers - 1]; ++j) {
		float sigmoidDerivative = a[layers - 1][j] * (1 - a[layers - 1][j]);
		err[layers - 1][j] = (a[layers - 1][j] - y[j]) * sigmoidDerivative;
	}

	/* Error backpropagation */
	for (int i = layers - 2; i >= 1; --i) {
		int m = neuron_num[i + 1];
		int n = neuron_num[i];

		#pragma omp parallel for num_threads(thread_count)
		for (int k = 0; k < n; ++k) {
			float temp = 0;
			for (int j = 0; j < m; ++j)
				temp += W[i + 1][j * n + k] * err[i + 1][j];

			float sigmoidDerivative = a[i][k] * (1 - a[i][k]);
			err[i][k] = temp * sigmoidDerivative;
		}
	}
}

float ANN::getLoss(vector<float> &y) const
{
	float loss = 0;
	for (int j = 0; j < neuron_num[layers - 1]; ++j)
		loss += 0.5 * pow(y[j] - a[layers - 1][j], 2.0);
	return loss;
}

void ANN::train(vector< vector<float> > &X, vector< vector<float> > &Y)
{
	random_device rd;
	mt19937 g(rd());
	mt19937 g_copy = g;
	shuffle(X.begin(), X.end(), g);
	shuffle(Y.begin(), Y.end(), g_copy);

	int s; // number of mini-batches
	int lastBatchSize;
	int remainder = X.size() % minibatchSize;
	if (remainder == 0) {
		s = X.size() / minibatchSize;
		lastBatchSize = minibatchSize;
	} else {
		s = X.size() / minibatchSize + 1;
		lastBatchSize = remainder;
	}

	cout << "Training in progress...\n";
	bool coin = false;
	double start, end;
	start = omp_get_wtime();

	for (int epoch = 1; epoch <= epochs; ++epoch) {
		/* for each epoch */
		if (!coin && epoch > 30) {
			coin = !coin;
			setLearningRate(0.5);
		}

		float avgLoss = 0;
		double start1, end1;
		start1 = omp_get_wtime();
		
		for (int batch = 0; batch < s; ++batch) {
			/* for each mini-batch */

			/* initialize gradient and biasGradient */
			float **gradient = new float *[layers];
			float **biasGradient = new float *[layers];

			for (int i = 1; i < layers; ++i) {
				int m = neuron_num[i];
				int n = neuron_num[i - 1];

				gradient[i] = new float[m * n];
				for (int j = 0; j < m * n; ++j)
					gradient[i][j] = 0;

				biasGradient[i] = new float[m];
				for (int j = 0; j < m; ++j)
					biasGradient[i][j] = 0;
			}
			// initialize gradient and biasGradient

			for (int i = batch * s; i < (batch + 1) * s && i < X.size(); ++i) {
				/* for each sample data in the mini-batch */
				feedForward(X[i]);
				backpropagate(Y[i]);
				avgLoss += getLoss(Y[i]);

				for (int l = layers - 1; l >= 1; --l) {
					int m = neuron_num[l];
					int n = neuron_num[l - 1];

					#pragma omp parallel for num_threads(thread_count)
					for (int j = 0; j < m; ++j) {
						for (int k = 0; k < n; ++k)
							gradient[l][j * n + k] += a[l - 1][k] * err[l][j];
						biasGradient[l][j] += err[l][j];
					}
				}
			} // for each sample data in the mini-batch

			int size = (batch == s - 1) ? lastBatchSize : minibatchSize;
			for (int l = layers - 1; l >= 1; --l) {
				int m = neuron_num[l];
				int n = neuron_num[l - 1];

				#pragma omp parallel for num_threads(thread_count)
				for (int j = 0; j < m; ++j) {	
					for (int k = 0; k < n; ++k)
						gradient[l][j * n + k] /= size;
					biasGradient[l][j] /= size;
				}
			}

			for (int l = layers - 1; l >= 1; --l) {
				int m = neuron_num[l];
				int n = neuron_num[l - 1];

				#pragma omp parallel for num_threads(thread_count)
				for (int j = 0; j < m; ++j) {
					for (int k = 0; k < n; ++k)
						W[l][j * n + k] -= r * gradient[l][j * n + k];
					b[l][j] -= r * biasGradient[l][j];
				}
			}
		} // for each mini-batch

		end1 = omp_get_wtime();
		avgLoss /= X.size();
		cout << "Average loss of epoch #" << epoch << ": "
			<< avgLoss << endl;
		cout << "Training time: " << (end1 - start1) << " s" << endl;
	} // for each epoch

	end = omp_get_wtime();
	cout << "Total training time: " << (end - start) << " s" << endl;
	cout << "Average training time per epoch: " << (end - start) / epochs
		<< " s" << endl;
}

vector<float> ANN::predict(vector<float> &x)
{
	feedForward(x);
	vector<float> p;
	for (int j = 0; j < neuron_num[layers - 1]; ++j)
		p.push_back(a[layers - 1][j]);
	return p;
}

void ANN::storeWeights(const string &filename) const
{
	ofstream ofs(filename);
	
	for (int i = 1; i < layers; ++i) {
		int m = neuron_num[i];
		int n = neuron_num[i - 1];

		for (int j = 0; j < m; ++j) {
			for (int k = 0; k < n; ++k)
				ofs << W[i][j * n + k] << ' ';
			ofs << '\n';
		}
		ofs << '\n';
	}

	for (int i = 1; i < layers; ++i) {
		for (int j = 0; j < neuron_num[i]; ++j)
			ofs << b[i][j] << ' ';
		ofs << '\n';
	}

	ofs.close();
}

void ANN::loadWeights(const string &filename)
{
	ifstream ifs(filename);

	for (int i = 1; i < layers; ++i) {
		int m = neuron_num[i];
		int n = neuron_num[i - 1];

		for (int j = 0; j < m; ++j)
			for (int k = 0; k < n; ++k)
				ifs >> W[i][j * n + k];
	}

	for (int i = 1; i < layers; ++i)
		for (int j = 0; j < neuron_num[i]; ++j)
			ifs >> b[i][j];

	ifs.close();
}

float ANN::sigmoid(float x)
{
	return 1.0 / (1.0 + exp(-x));
}
