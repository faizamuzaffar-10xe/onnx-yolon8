
import onnx
from onnx import helper, ValueInfoProto

def extract_subgraph(original_model, output_names, new_model_name):
    # Create a new model with the specified outputs
    graph = original_model.graph
    
    # Find the nodes that produce the specified outputs
    output_value_infos = []
    for output_name in output_names:
        for value_info in graph.value_info:
            if value_info.name == output_name:
                output_value_infos.append(value_info)
                break
        else:
            # If not found in value_info, check graph outputs
            for output in graph.output:
                if output.name == output_name:
                    output_value_infos.append(output)
                    break
            else:
                raise ValueError(f"Output {output_name} not found in model")
    
    # Create a new graph with the specified outputs
    new_graph = helper.make_graph(
        nodes=graph.node,
        name=graph.name + "_subgraph",
        inputs=graph.input,
        outputs=output_value_infos,
        initializer=graph.initializer,
        value_info=graph.value_info
    )
    
    # Create the new model
    new_model = helper.make_model(new_graph, producer_name="onnx-subgraph-extractor")
    new_model.opset_import.extend(original_model.opset_import)
    
    # Save the new model
    onnx.save(new_model, new_model_name)
    return new_model

# Load the original model
original_model = onnx.load('yolov8n.onnx')

# Extract first subgraph (up to specified intermediate outputs)
intermediate_outputs = [
    '/model.22/Concat_output_0',
    '/model.22/dfl/conv/Conv_output_0',
    '/model.22/Sigmoid_output_0'
]
first_subgraph = extract_subgraph(original_model, intermediate_outputs, 'yolov8n_first_subgraph.onnx')

# Extract second subgraph (from intermediate outputs to final outputs)
# We need to modify the original graph to use the intermediate outputs as inputs
def extract_second_subgraph(original_model, intermediate_outputs, new_model_name):
    graph = original_model.graph
    
    # Create new inputs from the intermediate outputs
    new_inputs = []
    for output_name in intermediate_outputs:
        # Find the type info for this output
        for value_info in graph.value_info:
            if value_info.name == output_name:
                new_input = ValueInfoProto()
                new_input.CopyFrom(value_info)
                new_input.name = output_name
                new_inputs.append(new_input)
                break
        else:
            # If not found in value_info, check graph outputs
            for output in graph.output:
                if output.name == output_name:
                    new_input = ValueInfoProto()
                    new_input.CopyFrom(output)
                    new_input.name = output_name
                    new_inputs.append(new_input)
                    break
            else:
                raise ValueError(f"Output {output_name} not found in model")
    
    # Find all nodes that depend on these intermediate outputs
    dependent_nodes = []
    for node in graph.node:
        # Check if any of the node's inputs are in our intermediate outputs
        if any(inp in intermediate_outputs for inp in node.input):
            dependent_nodes.append(node)
    
    # Create a new graph with these nodes
    new_graph = helper.make_graph(
        nodes=dependent_nodes,
        name=graph.name + "_second_subgraph",
        inputs=new_inputs,
        outputs=graph.output,
        initializer=[],  # We assume all needed initializers are in the first subgraph
        value_info=graph.value_info
    )
    
    # Create the new model
    new_model = helper.make_model(new_graph, producer_name="onnx-subgraph-extractor")
    new_model.opset_import.extend(original_model.opset_import)
    
    # Save the new model
    onnx.save(new_model, new_model_name)
    return new_model

second_subgraph = extract_second_subgraph(original_model, intermediate_outputs, 'yolov8n_second_subgraph.onnx')
