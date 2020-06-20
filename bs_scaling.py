from fastai2.basics import *
from fastai2.vision.all import *
from fastai2.callback.all import *
from fastai2.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
from fastai2.vision.models.xresnet import *
from fastai2.callback.mixup import *
from fastscript import *

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80


def get_dbunch(size, woof, bs, sh=0., workers=None):
    if size <= 224:
        path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else:
        path = URLs.IMAGEWOOF if woof else URLs.IMAGENETTE
    source = untar_data(path)
    if workers is None: workers = min(8, num_cpus())
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter=GrandparentSplitter(valid_name='val'),
                       get_items=get_image_files, get_y=parent_label)
    item_tfms = [RandomResizedCrop(size, min_scale=0.35), FlipItem(0.5)]
    batch_tfms = RandomErasing(p=0.9, max_count=3, sh=sh) if sh else None
    return dblock.databunch(source, path=source, bs=bs, num_workers=workers,
                            item_tfms=item_tfms, batch_tfms=batch_tfms)


@call_parse
def main(
        gpu: Param("GPU to run on", int) = None,
        woof: Param("Use imagewoof (otherwise imagenette)", int) = 0,
        lr: Param("Learning rate", float) = 1e-2,
        size: Param("Size (px: 128,192,256)", int) = 128,
        sqrmom: Param("sqr_mom", float) = 0.99,
        mom: Param("Momentum", float) = 0.9,
        eps: Param("epsilon", float) = 1e-6,
        epochs: Param("Number of epochs", int) = 5,
        bs: Param("Batch size", int) = 64,
        mixup: Param("Mixup", float) = 0.,
        opt: Param("Optimizer (adam,rms,sgd,ranger)", str) = 'ranger',
        arch: Param("Architecture", str) = 'xresnet50',
        sh: Param("Random erase max proportion", float) = 0.,
        sa: Param("Self-attention", int) = 0,
        sym: Param("Symmetry for self-attention", int) = 0,
        beta: Param("SAdam softplus beta", float) = 0.,
        act_fn: Param("Activation function", str) = 'MishJit',
        fp16: Param("Use mixed precision training", int) = 0,
        pool: Param("Pooling method", str) = 'AvgPool',
        dump: Param("Print model; don't train", int) = 0,
        runs: Param("Number of times to repeat training", int) = 1,
        wd: Param("Weight decay", int) = 1e-2,
        meta: Param("Metadata (ignored)", str) = '',
        ):
    """Distributed training of Imagenette."""

    # gpu = setup_distrib(gpu)
    if gpu is not None: torch.cuda.set_device(gpu)
    if opt == 'adam':
        opt_func = partial(Adam, mom=mom, sqr_mom=sqrmom, eps=eps)
    elif opt == 'rms':
        opt_func = partial(RMSprop, sqr_mom=sqrmom)
    elif opt == 'sgd':
        opt_func = partial(SGD, mom=mom)
    elif opt == 'ranger':
        opt_func = partial(ranger, mom=mom, sqr_mom=sqrmom, eps=eps, beta=beta)

    dbunch = get_dbunch(size, woof, bs, sh=sh)
    if not gpu: print(f'lr: {lr}; size: {size}; sqrmom: {sqrmom}; mom: {mom}; eps: {eps}')

    m, act_fn, pool = [globals()[o] for o in (arch, act_fn, pool)]

    for run in range(runs):
        print(f'Run: {run}')
        learn = Learner(dbunch, m(c_out=10, act_cls=act_fn, sa=sa, sym=sym, pool=pool), opt_func=opt_func, \
                        metrics=[accuracy, top_k_accuracy], loss_func=LabelSmoothingCrossEntropy())
        if dump: print(learn.model); exit()
        if fp16: learn = learn.to_fp16()
        cbs = MixUp(mixup) if mixup else []
        # n_gpu = torch.cuda.device_count()
        # if gpu is None and n_gpu: learn.to_parallel()
        if num_distrib() > 1: learn.to_distributed(gpu)  # Requires `-m fastai.launch`
        learn.fit_flat_cos(epochs, lr, wd=wd, cbs=cbs)
    return learn


root_dir = Path("scaling")
os.makedirs(root_dir, exist_ok=True)


def get_mom(mom, scale):
    return np.exp(np.log(0.5) / (np.log(0.5) / np.log(mom) / scale))


BS = 64
SQ_MOM = 0.99
MOM = 0.95
LR = 8e-3
WD = 1e-2

if not os.path.exists(root_dir / 'results.csv'):
    with open(root_dir / 'results.csv', 'a') as f:
        f.write("bs,lr,wd,mom,sq_mom,loss,acc\n")

for i in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    if i == 0:
        name = "all_lin"
        scaling = "lin"
        scale_lr = True
        scale_wd = True
        scale_mom = True
        scale_sqmom = True
    elif i == 1:
        name = "all_sqrt"
        scaling = "sqrt"
        scale_lr = True
        scale_wd = True
        scale_mom = True
        scale_sqmom = True
    elif i == 2:
        name = "lr_lin"
        scaling = "lin"
        scale_lr = True
        scale_wd = False
        scale_mom = False
        scale_sqmom = False
    elif i == 3:
        name = "wd_lin"
        scaling = "lin"
        scale_lr = False
        scale_wd = True
        scale_mom = False
        scale_sqmom = False
    elif i == 4:
        name = "moms_lin"
        scaling = "lin"
        scale_lr = False
        scale_wd = False
        scale_mom = True
        scale_sqmom = True
    elif i == 5:
        name = "mom_lin"
        scaling = "lin"
        scale_lr = False
        scale_wd = False
        scale_mom = True
        scale_sqmom = False
    elif i == 6:
        name = "sqmom_lin"
        scaling = "lin"
        scale_lr = False
        scale_wd = False
        scale_mom = False
        scale_sqmom = True
    elif i == 7:
        name = "not_lr_lin"
        scaling = "lin"
        scale_lr = False
        scale_wd = True
        scale_mom = True
        scale_sqmom = True
    elif i == 8:
        name = "not_wd_lin"
        scaling = "lin"
        scale_lr = True
        scale_wd = False
        scale_mom = True
        scale_sqmom = True
    elif i == 9:
        name = "not_moms_lin"
        scaling = "lin"
        scale_lr = True
        scale_wd = True
        scale_mom = False
        scale_sqmom = False
    else:
        name = "no"
        scaling = None
        scale_lr = False
        scale_wd = False
        scale_mom = False
        scale_sqmom = False

    with open(root_dir / (name + '.csv'), 'a') as f:
        f.write("," + ",".join(list(8e-3 * np.power([1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16],
                0 if scaling is None else 1 if scaling == "lin" else 1/2))) + "\n")

    for bs in (8, 16, 32, 64, 128, 256):
        with open(root_dir / (name + '.csv'), 'a') as f:
            f.write(str(bs) + ",")
            for j in (16, 32, 128, 256):
                f.write(",")
        for scale_offset in (1/4, 1/2, 1, 2, 4):
            if scale_offset < 1/3 and bs < 64 or scale_offset > 3 and bs > 64:
                with open(root_dir / (name + '.csv'), 'a') as f:
                    f.write(",")
                continue
            if scaling is not None or scale_offset < 1/3:
                if 'learn' in locals():
                    learn.destroy()
                torch.cuda.empty_cache()
                if scaling == "lin":
                    scale = bs / BS * scale_offset
                elif scaling == "sqrt":
                    scale = np.sqrt(bs / BS * scale_offset)
                else:
                    scale = 1
                sq_mom = get_mom(SQ_MOM, scale) if scale_sqmom else SQ_MOM
                mom = get_mom(MOM, scale) if scale_mom else MOM
                lr = LR * scale if scale_lr else LR
                wd = WD * scale if scale_wd else WD
                learn = main(lr=lr, sqrmom=sq_mom, mom=mom, epochs=20, bs=bs, )
                # data = get_dbunch(128, 0, bs)
                # learn = cnn_learner(data, xse_resnext50, pretrained=False, metrics=[accuracy]).to_fp16()
                # learn.fit_one_cycle(20, lr, moms=(sq_mom, mom), wd=wd)
                results = learn.validate(metrics=[accuracy])
                text = ",".join([str(bs), str(lr), str(wd), str(mom), str(sq_mom), str(results[0]), str(float(results[1]))])
                print(text)
                with open(root_dir / 'results.csv', 'a') as f:
                    f.write(text + "\n")
            with open(root_dir / (name + '.csv'), 'a') as f:
                f.write(results[0] + ",")
        with open(root_dir / (name + '.csv'), 'a') as f:
            f.write("\n")
