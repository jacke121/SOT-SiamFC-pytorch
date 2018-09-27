from collections import OrderedDict

opts = OrderedDict()

opts['lr'] = 0.0001
opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10
opts['ft_layers'] = ['conv', 'fc']
opts['lr_mult'] = {'fc': 10}
opts['n_cycles'] = 50