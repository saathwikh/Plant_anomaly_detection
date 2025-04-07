# grad_cam_utils.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models
from PIL import Image

def generate_grad_cam(model, input_tensor, class_idx, orig_image):
    model.eval()
    
    # Get the last conv layer
    final_conv = model.layer4[-1].conv3
    
    gradients = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    hook_grad = final_conv.register_backward_hook(save_gradient)
    hook_act = final_conv.register_forward_hook(save_activation)

    # Forward + backward
    output = model(input_tensor)
    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    # Gradients and activations
    grads_val = gradients[0].cpu().data.numpy()[0]
    act_val = activations[0].cpu().data.numpy()[0]

    # Global Average Pooling
    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(act_val.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act_val[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (512, 512))
    cam -= np.min(cam)
    cam /= np.max(cam)
    cam = np.uint8(255 * cam)

    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    orig = np.array(orig_image.resize((512, 512)))
    if orig.shape[2] == 1:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    
    overlay = 0.5 * cam + 0.5 * orig
    output_path = "static/cam_output.jpg"
    cv2.imwrite(output_path, overlay)

    # Remove hooks
    hook_grad.remove()
    hook_act.remove()

    return "/cam"
