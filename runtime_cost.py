import torch
from torch.profiler import profile, record_function, ProfilerActivity
from builder.builder import Trainer


# Execution time
dummy_input = torch.randn(1, 3, 256, 256).cuda()
model = Trainer(model_name='xception').cuda().eval()
# model = Trainer(model_name='xception', use_mc=True, use_aim=True).cuda()

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for i in range(20):
        model(dummy_input)


print(prof.key_averages().table(sort_by="cpu_time_total"))
