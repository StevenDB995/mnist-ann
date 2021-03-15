#ifndef ANN_H
#define ANN_H

#include <vector>

class ANN
{
public:
	ANN(int layers);
	void setLayers(int layers); // set number of layers
	void setNeurons(int index, int neurons); // set number of neurons at specified layer
	void build(void);

	void setLearningRate(float r);
	void setMinibatchSize(int size);
	void setEpochs(int epochs);
	void set_num_threads(int threads);

	void train(std::vector< std::vector<float> > &X, std::vector< std::vector<float> > &Y);
	std::vector<float> predict(std::vector<float> &x);
	void storeWeights(const std::string &filename) const;
	void loadWeights(const std::string &filename);
		
private:
	int layers; // number of layers
	int *neuron_num; // neuron_num[i]: number of neurons at the ith layer

	float **a; // activations
	float **b; // bias
	float **err; // errors
	float **W; // weights

	float r; // learning rate
	int minibatchSize; // mini-batch size
	int epochs; // number of epochs
	int thread_count; // number of threads

	void feedForward(std::vector<float> &x);
	void backpropagate(std::vector<float> &y);
	float getLoss(std::vector<float> &y) const;
	
	static float sigmoid(float x);
};

#endif
