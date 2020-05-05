import torch
import sys
import os.path


# to add above path so that we can import built libraries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                os.path.pardir)))

model = torch.load("../models/model.pkl")
model.eval()
example = torch.rand(10)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("../models/model_script.pt")


output = traced_script_module(torch.ones(10))
print(output)
