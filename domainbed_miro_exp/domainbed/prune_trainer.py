PRUNED_NUMs = {}
# For PACS
# PRUNE_RATIO = 0.1
# For OfficeHome
PRUNE_RATIO = 0.05

import collections
import json
import time
import copy
from pathlib import Path

from domainbed.masking import *
import numpy as np
import torch
import torch.utils.data
from collections import defaultdict, OrderedDict

from domainbed.datasets import get_dataset, split_dataset
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import swad as swad_module
# *************************************************************************************** #
def get_num_params(encoder):
    active = 0
    total = 0
    for k, v in encoder.named_modules():
        if hasattr(v, 'weight_mask'):
    #         print(v.weight_mask.shape)
            active += float(torch.sum(v.weight_mask != 0))
            total += v.weight_mask.numel()
        if hasattr(v, 'bias_mask'):
    #         print(v.bias_mask.shape)
            active += float(torch.sum(v.bias_mask != 0))
            total += v.bias_mask.numel()
    return active, total
# *************************************************************************************** #
class DSS:
    def __init__(self):
        pass
        # mode 0: dot products of normalized
        # mode 1: simple mean l2
    def compute(self, feat_source, feat_target, mode=0):
        # B, C, H, W
        B, C, H, W = feat_source.shape
        if mode == 0:
            v1 = feat_source.view(B, C, -1) 
            v1 = v1 / torch.linalg.norm(v1, dim=-1).unsqueeze(-1)
            v2 = feat_target.view(B, C, -1) 
            v2 = v2 / torch.linalg.norm(v2, dim=-1).unsqueeze(-1)
            prod = (v1 * v2).sum(dim=-1)
            # Larger prod means more similar
            return prod
        elif mode == 1:
            return torch.mean(torch.sqrt((feat_source - feat_target) ** 2), [2, 3])
#         elif mode == 2:
#             return ssim(feat_source, feat_target)
# *************************************************************************************** #

def add_masking_hooks(model):
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            add_attr_masking(layer, 'weight', ParameterMaskingType.Soft, True)
            if hasattr(layer, "bias") and layer.bias is not None:
                add_attr_masking(layer, 'bias', ParameterMaskingType.Soft, True)
# *************************************************************************************** #

def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def compute_score(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None):
    logger.info("")
    dss = DSS()
    #######################################################
    # setup dataset & loader
    #######################################################
    args.real_test_envs = test_envs  # for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    test_splits = []

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs))
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=np.int)

    batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()

    logger.info(f"Batch sizes for each domain: {batch_sizes} (total={sum(batch_sizes)})")

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / batch_size
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epoch = min(steps_per_epochs)
    # epoch is computed by steps_per_epoch
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    logger.info(f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}")

    # setup loaders
    # train_loaders = [
    #     InfiniteDataLoader(
    #         dataset=env,
    #         weights=env_weights,
    #         batch_size=batch_size,
    #         num_workers=dataset.N_WORKERS,
    #     )
    #     for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
    # ]
    # print("length of train loaders", sum([len(x) for x in train_loaders]))
    
    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
        batchsize = hparams["test_batchsize"]
        loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
        if args.prebuild_loader:
            loader_kwargs = FastDataLoader(**loader_kwargs)
        eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))

    loaders = {}
    for name, loader_kwargs, weights in eval_meta:
        # env\d_[in|out]
        env_name, inout = name.split("_")
        env_num = int(env_name[3:])

        if isinstance(loader_kwargs, dict):
            test_loader = FastDataLoader(**loader_kwargs)
        elif isinstance(loader_kwargs, FastDataLoader):
            test_loader = loader_kwargs
        if "in" in name:
            loaders[name] = test_loader
    test_env = test_envs[0]
    test_loader = loaders[f"env{test_env}_in"]
    del loaders[f"env{test_env}_in"]
    train_loaders = list(loaders.values())

    # print(loaders.keys())
    # print(test_envs)
    # exit()

    #######################################################
    # setup algorithm (model)
    #######################################################
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )

    algorithm.cuda()

    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    checkpoint_vals = collections.defaultdict(lambda: [])
    
    # Load Pretrained
    dataset_name = args.dataset
    print(dataset_name)
    ckpt = torch.load(f"train_output/{dataset_name}/TE{int(test_env)}_final.pth")
    algorithm.load_state_dict(ckpt['model_dict'])
    algorithm.register_fmap_hooks()
    algorithm.eval()

    scores = defaultdict(list)
    avg_src_fmaps = {}
    avg_tgt_fmaps = {}
    with torch.no_grad():
        count = 0
        for train_loader in train_loaders:
            for i, batch in enumerate(train_loader):
                # if i == 2:
                #     break
                train_x = batch["x"].cuda()
                train_y = batch["y"].cuda()
                logits = algorithm.predict(train_x)
                # for key, val in algorithm.selected_out.items():
                #     print(key, val.shape)
                for name, fmap in algorithm.selected_out.items():
                    if fmap.shape[0] < 128:
                        continue
                    if name not in avg_src_fmaps:
                        avg_src_fmaps[name] = fmap.cpu()
                    else: 
                        avg_src_fmaps[name] += fmap.cpu()
                count += 1
                algorithm.selected_out = OrderedDict()
        logger.info(count)
        avg_src_fmaps = {k:v/count for k,v in avg_src_fmaps.items()}
        avg_src_fmaps = {k:torch.mean(v, dim=0) for k,v in avg_src_fmaps.items()}

        count = 0
        for i, batch in enumerate(test_loader):
            # if i == 2:
            #     break
            test_x = batch["x"].cuda()
            test_y = batch["y"].cuda()
            logits = algorithm.predict(test_x)
            # for key, val in algorithm.selected_out.items():
            #     print(key, val.shape)
            for name, fmap in algorithm.selected_out.items():
                if fmap.shape[0] < 128:
                    continue
                if name not in avg_tgt_fmaps:
                    avg_tgt_fmaps[name] = fmap.cpu()
                else: 
                    avg_tgt_fmaps[name] += fmap.cpu()

            algorithm.selected_out = OrderedDict()
            count += 1

        logger.info(count)
        avg_tgt_fmaps = {k:v/count for k,v in avg_tgt_fmaps.items()}
        avg_tgt_fmaps = {k:torch.mean(v, dim=0) for k,v in avg_tgt_fmaps.items()}
        
    scores = {}
    scores1 = {}
    for name in avg_src_fmaps.keys():
        scores[name] = dss.compute(avg_src_fmaps[name].unsqueeze(0), avg_tgt_fmaps[name].unsqueeze(0), mode=0).squeeze() #B, C
        scores[name] = torch.nan_to_num(scores[name], 0)
        scores1[name] = dss.compute(avg_src_fmaps[name].unsqueeze(0), avg_tgt_fmaps[name].unsqueeze(0), mode=1).squeeze() #B, C
        scores1[name] = torch.nan_to_num(scores1[name], 0)
    
    logger.info("Score Computation Completed.")
    return scores, scores1

# ************************************************************************************************************************************ #
def prune_finetune(scores, test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None):
    logger.info("")

    #######################################################
    # setup dataset & loader
    #######################################################
    args.real_test_envs = test_envs  # for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    test_splits = []

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs))
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=np.int)

    batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()

    logger.info(f"Batch sizes for each domain: {batch_sizes} (total={sum(batch_sizes)})")

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / batch_size
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epoch = min(steps_per_epochs)
    # epoch is computed by steps_per_epoch
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    logger.info(f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}")

    # setup loaders
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size,
            num_workers=dataset.N_WORKERS,
        )
        for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]

    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
        batchsize = hparams["test_batchsize"]
        loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
        if args.prebuild_loader:
            loader_kwargs = FastDataLoader(**loader_kwargs)
        eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))

    #######################################################
    # setup algorithm (model)
    #######################################################
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )
    ###############################################################################################
    algorithm.cuda()
    # Load Pretrained
    test_env = test_envs[0]
    dataset_name = args.dataset
    print(dataset_name)
    ckpt = torch.load(f"train_output/{dataset_name}/TE{int(test_env)}_final.pth")

    algorithm.load_state_dict(ckpt['model_dict'])
    add_masking_hooks(algorithm.featurizer)
    ###############################################################################################
    active, total = get_num_params(algorithm.featurizer)
    print(f"{active} parameters out of {total}")
    logger.info(f"{active} parameters out of {total}\n\n")

    # ************************************************************************ #
    # Print before pruning but without logging into the records
    # this is performance of baseline
    evaluator = Evaluator(
        test_envs,
        eval_meta,
        n_envs,
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=target_env,
    )

    accuracies, summaries = evaluator.evaluate(algorithm)
    results = {}
    # results = (epochs, loss, step, step_time)
    results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
    # merge results
    results.update(summaries)
    results.update(accuracies)

    # print
    logger.info(misc.to_row(results_keys))
    last_results_keys = results_keys
    logger.info(misc.to_row([results[key] for key in results_keys]))
    # ************************************************************************ #
    for name, module in algorithm.featurizer.named_modules():
        # Skip layers
        if name in ["network.conv1"]:
            continue
        if name not in scores:
            continue
        score = scores[name]
        # print(name)
        val, idxs = torch.sort(torch.nan_to_num(torch.tensor(score), nan=-10))
        # val, idxs = torch.sort(torch.nan_to_num(torch.tensor(scores1[name]), nan=-10))
        idxs = idxs.tolist()
    #     idxs = idxs[::-1]
        # print(len(idxs))
        # print("For layer", name)
        # chn_num = module.weight_orig.data.shape[1]
        chn_num = module.weight_orig.data.shape[0]
        cur_prune_num = PRUNED_NUMs.get(name, 0)
        # print("Current Pruned Number", cur_prune_num)
        PRUNE_NUM = int((chn_num - cur_prune_num) * PRUNE_RATIO)
        PRUNE_NUM += cur_prune_num
        PRUNED_NUMs[name] = PRUNE_NUM
    #         print(idxs)
        # print(idxs[:PRUNE_NUM])
        # print(chn_num)
        mask = torch.ones(chn_num)
        mask[idxs[:PRUNE_NUM]] = 0
        set_mask(module, 'weight', mask.view(-1, 1, 1, 1).cuda())
        if hasattr(module, "bias") and module.bias is not None:
            set_mask(module, 'bias', mask.cuda())

    logger.info("Pruning Completed")
    active, total = get_num_params(algorithm.featurizer)
    print(f"{active} parameters out of {total}")
    logger.info(f"{active} parameters out of {total}\n\n")
    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)
    ###############################################################################################

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    #######################################################
    # start training loop
    #######################################################

    swad = None
    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm)
        swad_cls = getattr(swad_module, "LossValley")
        swad = swad_cls(evaluator, **hparams.swad_kwargs)

    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"

    for step in range(n_steps):
        step_start_time = time.time()
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(train_minibatches_iterator)
        # batches: {data_key: [env0_tensor, ...], ...}
        batches = misc.merge_dictlist(batches_dictlist)
        # to device
        batches = {
            key: [tensor.cuda() for tensor in tensorlist] for key, tensorlist in batches.items()
        }

        inputs = {**batches, "step": step}
        step_vals = algorithm.update(**inputs)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        if swad:
            # swad_algorithm is segment_swa for swad
            swad_algorithm.update_parameters(algorithm, step=step)

        if step % checkpoint_freq == 0:
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            accuracies, summaries = evaluator.evaluate(algorithm)
            results["eval_time"] = time.time() - eval_start_time

            # results = (epochs, loss, step, step_time)
            results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
            # merge results
            results.update(summaries)
            results.update(accuracies)

            # print
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys
            logger.info(misc.to_row([results[key] for key in results_keys]))
            records.append(copy.deepcopy(results))

            # update results to record
            results.update({"hparams": dict(hparams), "args": vars(args)})

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

            checkpoint_vals = collections.defaultdict(lambda: [])

            writer.add_scalars_with_prefix(summaries, step, f"{testenv_name}/summary/")
            writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")

            if args.model_save and step >= args.model_save:
                ckpt_dir = args.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)

                test_env_str = ",".join(map(str, test_envs))
                filename = "TE{}_{}.pth".format(test_env_str, step)
                if len(test_envs) > 1 and target_env is not None:
                    train_env_str = ",".join(map(str, train_envs))
                    filename = f"TE{target_env}_TR{train_env_str}_{step}.pth"
                path = ckpt_dir / filename

                save_dict = {
                    "args": vars(args),
                    "model_hparams": dict(hparams),
                    "test_envs": test_envs,
                    "model_dict": algorithm.cpu().state_dict(),
                }
                algorithm.cuda()
                if not args.debug:
                    torch.save(save_dict, path)
                else:
                    logger.debug("DEBUG Mode -> no save (org path: %s)" % path)

            # swad
            if swad:
                def prt_results_fn(results, avgmodel):
                    step_str = f" [{avgmodel.start_step}-{avgmodel.end_step}]"
                    row = misc.to_row([results[key] for key in results_keys if key in results])
                    logger.info(row + step_str)

                swad.update_and_evaluate(
                    swad_algorithm, results["train_out"], results["tr_outloss"], prt_results_fn
                )

                if hasattr(swad, "dead_valley") and swad.dead_valley:
                    logger.info("SWAD valley is dead -> early stop !")
                    break

                swad_algorithm = swa_utils.AveragedModel(algorithm)  # reset

        if step % args.tb_freq == 0:
            # add step values only for tb log
            writer.add_scalars_with_prefix(step_vals, step, f"{testenv_name}/summary/")

    # find best
    logger.info("---")
    records = Q(records)
    te_val_best = records.argmax("test_out")["test_in"]
    tr_val_best = records.argmax("train_out")["test_in"]
    last = records[-1]["test_in"]

    in_key = "train_out"
    tr_val_best_indomain = records.argmax("train_out")[in_key]
    last_indomain = records[-1][in_key]

    # NOTE for clearity, report only training-domain validation results.
    ret = {
        #  "test-domain validation": te_val_best,
        "training-domain validation": tr_val_best,
        #  "last": last,
        #  "last (inD)": last_indomain,
        #  "training-domain validation (inD)": tr_val_best_indomain,
    }

    # Evaluate SWAD
    if swad:
        swad_algorithm = swad.get_final_model()
        if hparams["freeze_bn"] is False:
            n_steps = 500 if not args.debug else 10
            logger.warning(f"Update SWAD BN statistics for {n_steps} steps ...")
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps)

        logger.warning("Evaluate SWAD ...")
        accuracies, summaries = evaluator.evaluate(swad_algorithm)
        results = {**summaries, **accuracies}
        start = swad_algorithm.start_step
        end = swad_algorithm.end_step
        step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
        row = misc.to_row([results[key] for key in results_keys if key in results]) + step_str
        logger.info(row)

        ret["SWAD"] = results["test_in"]
        ret["SWAD (inD)"] = results[in_key]

    for k, acc in ret.items():
        logger.info(f"{k} = {acc:.3%}")

    return ret, records

# ************************************************************************************************************************************ #
def prune_finetune_nonuni(scores, test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None):
    logger.info("")

    #######################################################
    # setup dataset & loader
    #######################################################
    args.real_test_envs = test_envs  # for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    test_splits = []

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs))
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=np.int)

    batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()

    logger.info(f"Batch sizes for each domain: {batch_sizes} (total={sum(batch_sizes)})")

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / batch_size
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epoch = min(steps_per_epochs)
    # epoch is computed by steps_per_epoch
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    logger.info(f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}")

    # setup loaders
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size,
            num_workers=dataset.N_WORKERS,
        )
        for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]

    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
        batchsize = hparams["test_batchsize"]
        loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
        if args.prebuild_loader:
            loader_kwargs = FastDataLoader(**loader_kwargs)
        eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))

    #######################################################
    # setup algorithm (model)
    #######################################################
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )
    ###############################################################################################
    algorithm.cuda()
    # Load Pretrained
    test_env = test_envs[0]
    # ckpt = torch.load(f"train_output/PACS/TE{int(test_env)}_final.pth")
    dataset_name = args.dataset
    print(dataset_name)
    ckpt = torch.load(f"train_output/{dataset_name}/TE{int(test_env)}_final.pth")
    algorithm.load_state_dict(ckpt['model_dict'])
    add_masking_hooks(algorithm.featurizer)
    ###############################################################################################
    active, total = get_num_params(algorithm.featurizer)
    print(f"{active} parameters out of {total}")
    logger.info(f"{active} parameters out of {total}\n\n")

    all_scores = []
    all_idxs = []
    masks = {}
    for name, module in algorithm.featurizer.named_modules():
        # Skip layers
        if name in ["network.conv1"]:
            continue
        if name not in scores:
            continue
        score = scores[name]
        # print(name)
        all_scores += score
        all_idxs += [(name, i) for i in range(len(score))]
        masks[name] = torch.ones(len(score))
    print(len(all_scores))
    print(len(all_idxs))
    val, idxs = torch.sort(torch.nan_to_num(torch.tensor(all_scores), nan=-10))
    idxs = idxs.tolist()
    name = "total"
    total_num = len(all_idxs)
    cur_prune_num = PRUNED_NUMs.get(name, 0)
    # print("Current Pruned Number", cur_prune_num)
    PRUNE_NUM = int((total_num - cur_prune_num) * PRUNE_RATIO)
    PRUNE_NUM += cur_prune_num
    PRUNED_NUMs[name] = PRUNE_NUM
    idx_names = all_idxs[:PRUNE_NUM]
    
    for name, idx in idx_names:
        masks[name][idx] = 0
    
    for name, module in algorithm.featurizer.named_modules():
        # Skip layers
        if name in ["network.conv1"]:
            continue
        if name not in scores:
            continue
        mask = masks[name]
        set_mask(module, 'weight', mask.view(-1, 1, 1, 1).cuda())
        if hasattr(module, "bias") and module.bias is not None:
            set_mask(module, 'bias', mask.cuda())

    logger.info("Pruning Completed")
    active, total = get_num_params(algorithm.featurizer)
    print(f"{active} parameters out of {total}")
    logger.info(f"{active} parameters out of {total}\n\n")
    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)
    ###############################################################################################

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    #######################################################
    # start training loop
    #######################################################
    evaluator = Evaluator(
        test_envs,
        eval_meta,
        n_envs,
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=target_env,
    )

    swad = None
    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm)
        swad_cls = getattr(swad_module, "LossValley")
        swad = swad_cls(evaluator, **hparams.swad_kwargs)

    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"

    for step in range(n_steps):
        step_start_time = time.time()
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(train_minibatches_iterator)
        # batches: {data_key: [env0_tensor, ...], ...}
        batches = misc.merge_dictlist(batches_dictlist)
        # to device
        batches = {
            key: [tensor.cuda() for tensor in tensorlist] for key, tensorlist in batches.items()
        }

        inputs = {**batches, "step": step}
        step_vals = algorithm.update(**inputs)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        if swad:
            # swad_algorithm is segment_swa for swad
            swad_algorithm.update_parameters(algorithm, step=step)

        if step % checkpoint_freq == 0:
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            accuracies, summaries = evaluator.evaluate(algorithm)
            results["eval_time"] = time.time() - eval_start_time

            # results = (epochs, loss, step, step_time)
            results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
            # merge results
            results.update(summaries)
            results.update(accuracies)

            # print
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys
            logger.info(misc.to_row([results[key] for key in results_keys]))
            records.append(copy.deepcopy(results))

            # update results to record
            results.update({"hparams": dict(hparams), "args": vars(args)})

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

            checkpoint_vals = collections.defaultdict(lambda: [])

            writer.add_scalars_with_prefix(summaries, step, f"{testenv_name}/summary/")
            writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")

            if args.model_save and step >= args.model_save:
                ckpt_dir = args.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)

                test_env_str = ",".join(map(str, test_envs))
                filename = "TE{}_{}.pth".format(test_env_str, step)
                if len(test_envs) > 1 and target_env is not None:
                    train_env_str = ",".join(map(str, train_envs))
                    filename = f"TE{target_env}_TR{train_env_str}_{step}.pth"
                path = ckpt_dir / filename

                save_dict = {
                    "args": vars(args),
                    "model_hparams": dict(hparams),
                    "test_envs": test_envs,
                    "model_dict": algorithm.cpu().state_dict(),
                }
                algorithm.cuda()
                if not args.debug:
                    torch.save(save_dict, path)
                else:
                    logger.debug("DEBUG Mode -> no save (org path: %s)" % path)

            # swad
            if swad:
                def prt_results_fn(results, avgmodel):
                    step_str = f" [{avgmodel.start_step}-{avgmodel.end_step}]"
                    row = misc.to_row([results[key] for key in results_keys if key in results])
                    logger.info(row + step_str)

                swad.update_and_evaluate(
                    swad_algorithm, results["train_out"], results["tr_outloss"], prt_results_fn
                )

                if hasattr(swad, "dead_valley") and swad.dead_valley:
                    logger.info("SWAD valley is dead -> early stop !")
                    break

                swad_algorithm = swa_utils.AveragedModel(algorithm)  # reset

        if step % args.tb_freq == 0:
            # add step values only for tb log
            writer.add_scalars_with_prefix(step_vals, step, f"{testenv_name}/summary/")

    # find best
    logger.info("---")
    records = Q(records)
    te_val_best = records.argmax("test_out")["test_in"]
    tr_val_best = records.argmax("train_out")["test_in"]
    last = records[-1]["test_in"]

    in_key = "train_out"
    tr_val_best_indomain = records.argmax("train_out")[in_key]
    last_indomain = records[-1][in_key]

    # NOTE for clearity, report only training-domain validation results.
    ret = {
        #  "test-domain validation": te_val_best,
        "training-domain validation": tr_val_best,
        #  "last": last,
        #  "last (inD)": last_indomain,
        #  "training-domain validation (inD)": tr_val_best_indomain,
    }

    # Evaluate SWAD
    if swad:
        swad_algorithm = swad.get_final_model()
        if hparams["freeze_bn"] is False:
            n_steps = 500 if not args.debug else 10
            logger.warning(f"Update SWAD BN statistics for {n_steps} steps ...")
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps)

        logger.warning("Evaluate SWAD ...")
        accuracies, summaries = evaluator.evaluate(swad_algorithm)
        results = {**summaries, **accuracies}
        start = swad_algorithm.start_step
        end = swad_algorithm.end_step
        step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
        row = misc.to_row([results[key] for key in results_keys if key in results]) + step_str
        logger.info(row)

        ret["SWAD"] = results["test_in"]
        ret["SWAD (inD)"] = results[in_key]

    for k, acc in ret.items():
        logger.info(f"{k} = {acc:.3%}")

    return ret, records