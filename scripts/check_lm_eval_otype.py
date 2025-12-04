import os
import yaml

# Define a constructor function for the !function tag
def function_constructor(loader, node):
    # In this example, we assume !function takes a string and returns it
    # You would implement your specific logic here to create the desired object
    return loader.construct_scalar(node)

# Add the constructor to the SafeLoader
yaml.SafeLoader.add_constructor('!function', function_constructor)

# Function to print output_type from YAML files in the directory
def print_output_types(path):
    # Traverse through all YAML files in the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        # Load the YAML file
                        data = yaml.safe_load(f)
                        # Print output_type if it exists
                        output_type = data.get('output_type')
                        if output_type:
                            print(f"{file}: {output_type}")
                        # else:
                        #     print(f"{file}: No 'output_type' found")
                    except yaml.YAMLError as e:
                        print(f"Error reading {file_path}: {e}")
                        print (f.readlines())


tasks = "winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,gsm8k,mathqa,race,lambada_openai,mmlu".split(",")
for task in tasks:
    path_to_yamls = f'lm-evaluation-harness/lm_eval/tasks/{task}'
    print_output_types(path_to_yamls)
