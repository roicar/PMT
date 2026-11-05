def copy_dict_to_device(data_dict, device):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            copy_dict_to_device(value, device)
        else:
            data_dict[key] = value.to(device)
    return data_dict