var gaussian = require('gaussian');
var fs = require('fs');
var math = require('mathjs');
var loader = require('./mnistloader');
var distr = gaussian(0,1);
var Neuron = function(bias, weights) {
	this.bias = bias;
	this.weights = weights;
	this.activate = function(input) {
		var dot = 0;
		for(var i=0;i<input.length;i++) {
			dot+=this.weights[i]*input[i];
		}
		if(isNaN(dot)) {
			dot = Number.MAX_VALUE;
		}
		return dot + this.bias;
	}
}
var CrossEntropyCost = function() {
	this.fn = function(activation, output) {
		var result = [];
		for(var i=0;i<activation.length;i++) {
			result[i] = (-output[i]*Math.log(activation[i]))-((1-output[i])*Math.log(1-activation[i]));
			if(isNaN(result[i])) {
				result[i] = 0;
			}
		}
		return result;
	}
	this.delta = function(activation, output) {
		var out = [];
		for(var i=0;i<activation.length;i++) {
			out[i] = activation[i] - output[i];
		}
		return out;
	}
}
var Network = function(sizes, cost) {
	this.num_layers = sizes.length;
	this.sizes = sizes;
	this.cost = cost;
	this.neurons = new Array(this.num_layers-1);
	for(var i=1;i<sizes.length;i++) {
		this.neurons[i-1] = new Array(sizes[i]);
		for(var j=0;j<sizes[i];j++) {
			var bias = distr.ppf(Math.random());
			var x = sizes[i-1];
			var weights = new Array(x);
			for(var k=0;k<x;k++) {
				weights[k] = distr.ppf(Math.random())/Math.sqrt(x);
			}
			this.neurons[i-1][j] = new Neuron(bias, weights);
		}
	}
	this.feedForward = function(a) {
		var output = a;
		for(var i=0;i<this.neurons.length;i++) {
			var vector = [];
			for(var j=0;j<this.neurons[i].length;j++) {
				var neuron = this.neurons[i][j];
				vector[j] = sigmoid(neuron.activate(output));
			}
			output = vector;
		}
		return output;
	}
	this.SGD = function(training_data, epochs, mini_batch_size, rate, lambda, test_data, monitor) {
		var num = training_data.length;
		var testnum = test_data.length;
		var evaluation_cost = [];
		var evaluation_accuracy = [];
		var training_cost = [];
		var training_accuracy = [];
		for(var i=0;i<epochs;i++) {
			shuffle(training_data);
			var mini_batches = [];
			for(var j=0;j<num;j+=mini_batch_size) {
				mini_batches.push(training_data.slice(j, j+mini_batch_size));
			}
			for(var k=0;k<mini_batches.length;k++) {
				this.update_mini_batch(mini_batches[k], rate, lambda, num);
			}
			if(test_data) {
				console.log("Epoch " + i + " completed");
				if(monitor) {
					//trainingcost = this.total_cost(training_data, lambda);
					//training_cost.push(trainingcost);
					//console.log("Cost on training data: " + trainingcost);
					//trainingaccuracy = this.accuracy(training_data, true);
					//training_accuracy.push(trainingaccuracy);
					//console.log("Accuracy on training data: " + trainingaccuracy +"/"+num);
					//evalcost = this.total_cost(testing_data, lambda, true);
					//evaluation_cost.push(evalcost);
					//console.log("Cost on evaluation data: " + evalcost);
					evalaccuracy = this.accuracy(test_data);
					evaluation_accuracy.push(evalaccuracy);
					console.log("Accuracy on evaluation data: " + evalaccuracy +"/"+testnum);

				}
			}
			else {
				console.log("Epoch " + i + " completed");
			}
		}
		return [evaluation_cost, evaluation_accuracy, training_cost, training_accuracy];
	}
	this.update_mini_batch = function(mini_batch, rate, lambda, num) {
		for(var j=0;j<this.neurons.length;j++) {
			for(var k=0;k<this.neurons[j].length;k++) {
				var neuron = this.neurons[j][k];
				neuron.gradb = 0;
				neuron.gradw = [];
				for(var i=0;i<neuron.weights.length;i++) {
					neuron.gradw[i] = 0;
				}
			}
		}
		for(var i=0;i<mini_batch.length;i++) {
			this.backprop(mini_batch[i][0],mini_batch[i][1]);
		}
		for(var j=0;j<this.neurons.length;j++) {
			for(var k=0;k<this.neurons[j].length;k++) {
				var neuron = this.neurons[j][k];
				neuron.bias = neuron.bias - (rate/mini_batch.length)*neuron.gradb;
				for(var i=0;i<neuron.weights.length;i++) {
					neuron.weights[i] = (1-(rate*lambda/num))*neuron.weights[i]-(rate/mini_batch.length)*neuron.gradw[i];
				}
			}
		}
	}
	this.backprop = function(input, output) {
		var activation = input;
		var activations = [input];
		var outs = [];
		for(var j=0;j<this.neurons.length;j++) {
			var newacts = [];
			outs[j] = [];
			for(var k=0;k<this.neurons[j].length;k++) {
				var neuron = this.neurons[j][k];
				var out = neuron.activate(activation);
				outs[j][k] = out;
				newacts[k] = sigmoid(out);
			}
			activation = newacts;
			activations.push(newacts);
		}
		var change = this.cost.delta(activations[activations.length-1], output);
		for(var i=0;i<change.length;i++) {
			var neuron = this.neurons[this.neurons.length-1][i];
			this.neurons[this.neurons.length-1][i].change = change[i];
			this.neurons[this.neurons.length-1][i].gradb += change[i];
			for(var j=0;j<this.neurons[this.neurons.length-1][i].weights.length;j++) {
				this.neurons[this.neurons.length-1][i].gradw[j] += change[i] * activations[activations.length-2][j];
			}	
		}
		for(var j=this.neurons.length-2;j>=0;j--) {
			for(var k=0;k<this.neurons[j].length;k++) {
				var neuron = this.neurons[j][k];
				var out = outs[j][k];
				var spv = sigmoidD(out);
				var change = 0;
				for(var i=0;i<this.neurons[j+1].length;i++) {
					change += this.neurons[j+1][i].weights[k]*this.neurons[j+1][i].change;
				}
				change*=spv;
				neuron.change = change;
				neuron.gradb = neuron.gradb + change;
				for(var i=0;i<neuron.weights.length;i++) {
					var index = this.neurons.length - j;
					neuron.gradw[i] = neuron.gradw[i] + change * activations[activations.length-index-1][i];
				}
			}
		}
	}
	this.accuracy = function(test_data, convert) {
		convert = convert || false
		if(!convert) {
			var correct = 0;
			for(var i=0;i<test_data.length;i++) {
				var x = test_data[i][0];
				var y = test_data[i][1];
				var result = this.feedForward(x);
				var max = result[0];
				var maxIndex = 0;
				for(var j=1;j<result.length;j++) {
					if(result[j]>max) {
						maxIndex = j;
						max = result[j];
					}
				}
				if(y==maxIndex) {
					correct++;
				}
			}
			return correct;
		}
		else {
			var correct = 0;
			for(var i=0;i<test_data.length;i++) {
				var x = test_data[i][0];
				var y = test_data[i][1];
				var result = this.feedForward(x);
				var max = result[0];
				var maxIndex = 0;
				for(var j=1;j<result.length;j++) {
					if(result[j]>max) {
						maxIndex = j;
						max = result[j];
					}
				}
				max = y[0];
				var maxIndexY = 0;
				for(var j=1;j<y.length;j++) {
					if(y[j]>max) {
						maxIndexY = j;
						max = y[j];
					}
				}
				if(maxIndexY==maxIndex) {
					correct++;
				}
			}
			return correct;
		}
	}
	this.total_cost = function(test_data, lambda, convert) {
		convert = convert || false;
		var cost = [0,0,0,0,0,0,0,0,0,0];
		for(var i=0;i<test_data.length;i++) {
			var x = test_data[i][0];
			var y = test_data[i][1];
			var activation = this.feedforward(x);
			if(convert) {
				var output = [];
			    for(var i=0;i<10;i++) {
			    	output[i] = 0;
			    }
			    output[outInt] = 1;
			    y = output;
			}
			var newcost = this.cost.fn(activation, y);
			for(var i=0;i<cost.length;i++) {
				cost[i]+=newcost[i]/test_data.length;
			}
		}
		for(var i=0;i<cost.length;i++) {

		}
		return cost;
	}
}
function sigmoid(x) {
	return 1.0/(1.0+Math.exp(-x));
}
function sigmoidD(x) {
	return sigmoid(x)*(1-sigmoid(x));
}
function shuffle(array) {
    var counter = array.length, temp, index;

    // While there are elements in the array
    while (counter > 0) {
        // Pick a random index
        index = Math.floor(Math.random() * counter);

        // Decrease counter by 1
        counter--;

        // And swap the last element with it
        temp = array[counter];
        array[counter] = array[index];
        array[index] = temp;
    }

    return array;
}
var network = new Network([3,4,3], new CrossEntropyCost());
console.log("Beginning...");
/*var training_data = [[[3,2,3],[0,0,1]]];
network.SGD(training_data, 10,1, 10.0,0.1,[[[3,2,3],2]],true);*/
loader.loadTraining(function(training_data) {
	loader.loadTesting(function(testing_data) {
		network.SGD(training_data, 100, 1, 0.01, 5.0, testing_data, true);
	});
});