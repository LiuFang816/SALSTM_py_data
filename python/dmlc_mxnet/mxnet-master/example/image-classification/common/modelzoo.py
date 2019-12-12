import os
from util import download_file

_base_model_url = 'http://data.mxnet.io/models/'
_default_model_info = {
    'imagenet1k-inception-bn': {'symbol':_base_model_url+'imagenet/inception-bn/Inception-BN-symbol.json',
                             'params':_base_model_url+'imagenet/inception-bn/Inception-BN-0126.params'},
    'imagenet1k-resnet-18': {'symbol':_base_model_url+'imagenet/resnet/18-layers/resnet-18-symbol.json',
                             'params':_base_model_url+'imagenet/resnet/18-layers/resnet-18-0000.params'},
    'imagenet1k-resnet-34': {'symbol':_base_model_url+'imagenet/resnet/34-layers/resnet-34-symbol.json',
                             'params':_base_model_url+'imagenet/resnet/34-layers/resnet-34-0000.params'},
    'imagenet1k-resnet-50': {'symbol':_base_model_url+'imagenet/resnet/50-layers/resnet-50-symbol.json',
                             'params':_base_model_url+'imagenet/resnet/50-layers/resnet-50-0000.params'},
    'imagenet1k-resnet-101': {'symbol':_base_model_url+'imagenet/resnet/101-layers/resnet-101-symbol.json',
                             'params':_base_model_url+'imagenet/resnet/101-layers/resnet-101-0000.params'},
    'imagenet1k-resnet-152': {'symbol':_base_model_url+'imagenet/resnet/152-layers/resnet-152-symbol.json',
                             'params':_base_model_url+'imagenet/resnet/152-layers/resnet-152-0000.params'},
    'imagenet1k-resnext-50': {'symbol':_base_model_url+'imagenet/resnext/50-layers/resnext-50-symbol.json',
                             'params':_base_model_url+'imagenet/resnext/50-layers/resnext-50-0000.params'},
    'imagenet1k-resnext-101': {'symbol':_base_model_url+'imagenet/resnext/101-layers/resnext-101-symbol.json',
                             'params':_base_model_url+'imagenet/resnext/101-layers/resnext-101-0000.params'},
    'imagenet11k-resnet-152': {'symbol':_base_model_url+'imagenet-11k/resnet-152/resnet-152-symbol.json',
                             'params':_base_model_url+'imagenet-11k/resnet-152/resnet-152-0000.params'},
    'imagenet11k-place365ch-resnet-152': {'symbol':_base_model_url+'imagenet-11k-place365-ch/resnet-152-symbol.json',
                                          'params':_base_model_url+'imagenet-11k-place365-ch/resnet-152-0000.params'},
    'imagenet11k-place365ch-resnet-50': {'symbol':_base_model_url+'imagenet-11k-place365-ch/resnet-50-symbol.json',
                                         'params':_base_model_url+'imagenet-11k-place365-ch/resnet-50-0000.params'},
}

def download_model(model_name, dst_dir='./', meta_info=None):
    if meta_info is None:
        meta_info = _default_model_info
    meta_info = dict(meta_info)
    if model_name not in meta_info:
        return (None, 0)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    meta = dict(meta_info[model_name])
    assert 'symbol' in meta, "missing symbol url"
    model_name = os.path.join(dst_dir, model_name)
    download_file(meta['symbol'], model_name+'-symbol.json')
    assert 'params' in meta, "mssing parameter file url"
    download_file(meta['params'], model_name+'-0000.params')
    return (model_name, 0)
