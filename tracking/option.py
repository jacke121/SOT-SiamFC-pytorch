from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True

opts['model_path'] = '../checkpoints/siamfc_neo.pth'

opts['img_size'] = 352
opts['padding'] = 1

opts['batch_pos'] = 10
opts['batch_neg'] = 10
opts['batch_neg_cand'] = 10
opts['batch_test'] = 10

