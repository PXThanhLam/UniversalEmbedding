import torch
import xcit_retrieval
import patchconv_retrieval

checkpoint = torch.load('model_checkpoint/vitl_clip_curface_datasetv3_freeze_all/checkpoint_model_epoch_0.pth', map_location='cpu')
checkpoint_model = checkpoint['model']
# del checkpoint_model['head.kernel']
# del checkpoint_model['head.t']
del checkpoint_model['head.kernel.weight']
print(checkpoint_model.keys())
torch.save(checkpoint_model,'model_checkpoint/ckpt_for_submit_clip_freezeallthentune23_adapsubarcface_d3_e0.pth')
