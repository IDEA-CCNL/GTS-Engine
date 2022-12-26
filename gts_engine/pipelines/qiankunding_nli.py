#encoding=utf-8

import os
import json
from torch.utils.data import DataLoader
from transformers import BertTokenizer, MegatronBertForMaskedLM

from gts_common.registry import PIPELINE_REGISTRY
from gts_common.pipeline_utils import download_model_from_huggingface, generate_common_trainer, load_args, save_args
from qiankunding.utils.tokenization import get_train_tokenizer
from qiankunding.dataloaders.nli.dataloader_UnifiedMC import TaskDataModelUnifiedMCForNLI, TaskDatasetUnifiedMCForNLI
from qiankunding.models.nli.bert_UnifiedMC import BertUnifiedMCForNLI
from qiankunding.utils.evaluation import SentencePairEvaluator
from gts_common.logs_utils import Logger

logger = Logger().get_log()


@PIPELINE_REGISTRY.register(suffix=__name__)
def train_pipeline(args):
    # save args
    args = save_args(args)
    
    """ write a traininig pipeline and return the checkpoint path of the best model """
    model_name = "Erlangshen-UniMC-MegatronBERT-1.3B-Chinese"
    # download pretrained model if not exists
    download_model_from_huggingface(args.pretrained_model_dir, model_name, model_class=MegatronBertForMaskedLM, tokenizer_class=BertTokenizer)
    # Set path to load pretrained model
    args.pretrained_model = os.path.join(args.pretrained_model_dir, model_name)
    # init tokenizer
    tokenizer = get_train_tokenizer(args=args)            
    tokenizer.save_pretrained(args.save_path)
    # init label
    # shutil.copyfile(os.path.join(args.data_dir, args.label_data), os.path.join(args.save_path, "label.json"))
    # init model
    data_model = TaskDataModelUnifiedMCForNLI(args, tokenizer)
    #加载模型
    model = BertUnifiedMCForNLI(args, tokenizer)
    trainer, checkpoint = generate_common_trainer(args, args.save_path)
    # training
    trainer.fit(model, data_model)
    #验证集效果最好的模型文件地址
    checkpoint_path = checkpoint.best_model_path
    
    if args.test_data:
        output_save_path = os.path.join(args.save_path, 'predictions/')
        if not os.path.exists(output_save_path):
            os.makedirs(output_save_path)

        # Evaluation
        logger.info("Load checkpoint from {}".format(checkpoint_path))
        model = BertUnifiedMCForNLI.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
        model.cuda()
        model.eval() 

        evaluator = SentencePairEvaluator(args, model, data_model, output_save_path)
        test_acc = evaluator.evaluation(mode='test', data_set="test")

        task_info = json.load(open(os.path.join(args.task_dir, "task_info.json")))
        task_info["test_acc"] = test_acc
        with open(os.path.join(args.task_dir, "task_info.json"), mode="w") as f:
                json.dump(task_info, f, indent=4)

@PIPELINE_REGISTRY.register(suffix=__name__)
def prepare_inference(save_path):
    # load args
    args = load_args(save_path)

    # load tokenizer
    logger.info("Load tokenizer from {}".format(os.path.join(save_path, "vocab.txt")))
    inference_tokenizer = BertTokenizer.from_pretrained(save_path)

    # load model
    checkpoint_path = os.path.join(save_path, "best_model.ckpt")
    inference_model = BertUnifiedMCForNLI.load_from_checkpoint(checkpoint_path, tokenizer=inference_tokenizer)
    inference_model.eval()
    inference_model = inference_model.cuda()

    id2label = {0:"entailment", 1:"contradiction", 2:"neutral"}
    data_model = TaskDataModelUnifiedMCForNLI(args, inference_tokenizer)
    
    inference_suite = {
        "tokenizer": inference_tokenizer,
        "model": inference_model,
        "data_model": data_model,
        "id2label": id2label,
        "args": args
    }
    return inference_suite

@PIPELINE_REGISTRY.register(suffix=__name__)
def inference(samples, inference_suite):
    # 加载数据
    inner_samples = []
    question = "根据这段话"

    for idx, sample in enumerate(samples):
        textb = sample["sentence2"]
        label2id = {v:k for k,v in inference_suite["id2label"].items()}

        choice = [f"可以推断出：{textb}",f"不能推断出：{textb}",f"很难推断出：{textb}"]  

        label = label2id[sample["label"]]
        inner_sample = {
            "id":idx,
            "texta": sample["sentence1"],
            "textb": sample["sentence2"],
            "question":question,
            "choice":choice,
            "label":label
        }
        inner_sample["answer"] = inner_sample["choice"][inner_sample["label"]]
        inner_samples.append(inner_sample)

    dataset = TaskDatasetUnifiedMCForNLI(
        data_path=None,
        args=inference_suite["args"],
        used_mask=False,
        tokenizer=inference_suite["tokenizer"],
        load_from_list=True,
        samples=inner_samples,
        is_test=True,
        unlabeled=True
    )

    
    dataloader = DataLoader(dataset, shuffle=False, 
        collate_fn=inference_suite["data_model"].collate_fn, \
        batch_size=inference_suite["args"].valid_batchsize)

    pred_labels = []
    pred_probs = []


    for batch in dataloader:
        logits, classify_probs, predicts, labels, sample_embeds = inference_suite["model"].predict(batch)

        final_predicts = list(np.argmax(classify_probs, axis=1))
        for predict, prob in zip(final_predicts, classify_probs):    
            pred_labels.append(inference_suite["id2label"][predict])
            pred_probs.append(prob.tolist())


    result = {'predictions':pred_labels, 'probabilities':pred_probs}
    return result
