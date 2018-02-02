#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;

int main(int argc, char* argv[])
{
	// initialize a tensorflow session
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok())
	{
		std::cerr << status.ToString() << std::endl;
		return 1;
	}
	else
	{
		std::cout << "tf session start ok" << std::endl;
	}

	// Read protobuf graph created in python
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), "D:/projects/Deep_Learning/tensorflow/graph_cxx_api/models/graph.pb", &graph_def);
	if (!status.ok())
	{
		std::cerr << status.ToString() << std::endl;
		return 1;
	}
	else
	{
		std::cout << "reading graph ok" << std::endl;
	}

	// Add the graph to session
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << std::endl;
		return 1;
	}
	else
	{
		std::cout << "add graph to session ok" << std::endl;
	}


	// setup inputs and outputs
	// The saved graph des not need any inputs since it specified default values, here will change the values for demonstraion
	Tensor a(DT_FLOAT, TensorShape());
	a.scalar<float>()() = 3.0;

	Tensor b(DT_FLOAT, TensorShape());
	b.scalar<float>()() = 2.0;

	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
		{ "a", a },
		{ "b", b },
	};

	// The session will initialize the outputs
	std::vector<tensorflow::Tensor> outputs;

	// Run the session, evaluating our "c" operation from the graph
	status = session->Run(inputs, { "c" }, {}, &outputs);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	// Grab the first output(we only evaluated one graph node : "c")
		// and convert the node to a scalar representation.
	auto output_c = outputs[0].scalar<float>();

	// Print the results
	std::cout << outputs[0].DebugString() << std::endl; // Tensor<type: float shape: [] values: 6>
	std::cout << output_c() << std::endl; // 6

	// Free any resources used by the session
	session->Close();

	return 0;
}