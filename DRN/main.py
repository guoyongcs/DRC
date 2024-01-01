import time
import copy
import torch
import utility
import data
import loss
from model import Model
from option import args
from checkpoint import Checkpoint
from trainer import Trainer


utility.set_seed(args.seed)
loader = data.Data(args)
checkpoint = Checkpoint(args)
model = Model(args, checkpoint)

if args.pruning:
    from channel_pruning import ChannelPruning
    experiment = ChannelPruning(args, model, loader)
    experiment.channel_selecton()
elif args.search:
    from channel_search import Search
    # args_search = utility.parse_search_args(args)
    model = Model(args, checkpoint)
    searcher = Search(args, model, loader)
    searcher.search()
elif args.finetune:
    from finetune import Finetuner
    args_ft = utility.parse_ft_args(args)
    model_ft = Model(args_ft, checkpoint)
    finetuner = Finetuner(args_ft, model, model_ft, loader)
    finetuner.finetune()
elif args.quantization:
    from quantization import Quantization
    args_qt = utility.parse_qt_args(args)
    model_qt = Model(args_qt, checkpoint)
    quant = Quantization(args_qt, model, model_qt, loader)
    quant.quantization()
else:
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        epoch_time = time.time()
        t.train()
        t.test()
        epoch_time = time.time() - epoch_time
        log = utility.remain_time(epoch_time, t.scheduler.last_epoch, args.epochs)
        print(log)

checkpoint.done()

