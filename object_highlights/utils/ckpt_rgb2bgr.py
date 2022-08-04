import torch
import copy
import argparse


def state_dict_rgb2bgr(state_dict, layer_key="module.encoder._conv_stem.weight"):
    new_state_dict = state_dict.copy()
    new_state_dict["state_dict"][layer_key][:,:3,:,:] = state_dict["state_dict"][layer_key][:,[2,1,0],:,:]
    return new_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True,  type=str)
    args = parser.parse_args()

    mydict = torch.load(args.input)
    torch.save(state_dict_rgb2bgr(mydict, layer_key="module.encoder._conv_stem.weight"), args.output)