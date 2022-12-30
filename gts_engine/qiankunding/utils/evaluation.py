import json
from time import clock_settime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from gts_common.logs_utils import Logger

logger = Logger().get_log()

def result_eval(y_true, y_pred, label_names):
    label_ids = [i for i in range(len(label_names))]

    try:
        confused_matrix = confusion_matrix(y_true=y_true,
                                        y_pred=y_pred,
                                        labels=label_ids).tolist()
    except:
        confused_matrix = None

    cm_dict = {
        'matrix': confused_matrix,
        'label_list': label_names
    }

    try:
        eval_report = classification_report(y_true,
                                            y_pred,
                                            target_names=label_names,
                                            digits=2,
                                            zero_division=0,
                                            output_dict=True)
    except:
        eval_report = None

    global_acc = accuracy_score(y_true, y_pred)

    eval_results = {}
    eval_results['global_acc'] = global_acc
    eval_results['label_result'] = eval_report
    eval_results['confusion_matrix'] = cm_dict

    logger.info("global_acc: {}".format(global_acc))

    return eval_results

class Evaluator(object):
    
    def __init__(self, args, model, data_model, save_path):
        super().__init__()
        self.args, self.model, self.data_model, self.save_path = args, model, data_model, save_path

        self.task_type = args.task_type
        self.label_classes = self.data_model.label_classes
        logger.info("label_classes",self.label_classes)
        with open(self.save_path+"label_classes.json", 'w') as f:
                result = json.dumps(self.label_classes,ensure_ascii=False)
                f.write(result+"\n")
        self.label_classes_reverse = {v:k for k,v in self.label_classes.items()}
        logger.info("label_classes_reverse",self.label_classes_reverse)


    def save_to_file(self, data_set, results, y_true, y_pred):
        if data_set=="unlabeled":
            with open(self.save_path+"unlabeled_set_predictions.json", 'w') as f:
                for result in results:
                    result = json.dumps(result,ensure_ascii=False)
                    f.write(result+"\n")
        elif data_set in ["val","test"]:
            with open(self.save_path+f"{data_set}_set_predictions.json", 'w') as f:
                for result in results:
                    result = json.dumps(result,ensure_ascii=False)
                    f.write(result+"\n")

            eval_results = result_eval(y_true, y_pred, label_names=self.label_classes)
            
            
            with open(self.save_path+f"{data_set}_set_confusion_matrix.json", "w") as f:
                json.dump(eval_results, f, indent=4, ensure_ascii=False)

        elif data_set=="train":
            with open(self.save_path+"labeled_set_predictions.json", 'w') as f:
                for result in results:
                    result = json.dumps(result,ensure_ascii=False)
                    f.write(result+"\n")


        logger.info("Evaluation file saved at {}".format(self.save_path))

    def inference(self, test_loader, data_set, threshold):
        results = []
        y_true = []
        y_pred = []

        for batch in tqdm(test_loader):

            logits, probs, predicts, labels, _ = self.model.predict(batch)
            
            if data_set in ["val","test"]:
                y_true += list(labels)
                y_pred += list(predicts)

            for idx, (predict,prob,logit) in enumerate(zip(predicts,probs,logits)):
                
                pred = {
                # "id": int(batch["id"][idx]),
                'content': batch['sentence'][idx],
                'label': self.label_classes_reverse[predict],
                # 'content': batch['sentence'][idx],
                "probs": prob.tolist(),
            }
                if data_set=="unlabeled":
                    if max(prob.tolist()) > threshold:
                        results.append(pred)
                else:
                    results.append(pred)
        return results, y_true, y_pred

    def evaluation(self, mode, data_set, threshold=0):
        self.data_model.setup(mode)
        tokenizer = self.data_model.tokenizer
        
        if data_set=="unlabeled":
            test_loader = self.data_model.unlabeled_dataloader()
        elif data_set=="val":
            test_loader = self.data_model.val_dataloader()
        elif data_set=="test":
            test_loader = self.data_model.test_dataloader()
        elif data_set=="train":
            test_loader = self.data_model.train_dataloader()
            
        results, y_true, y_pred = self.inference(test_loader=test_loader, data_set=data_set, threshold=threshold)
        self.save_to_file(data_set, results, y_true, y_pred)
        acc = None
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
        return acc

class SentencePairEvaluator(Evaluator):
    def inference(self, test_loader, data_set, threshold):
        if self.task_type == "similarity":
            id2label = {0:0, 1:1}
        elif self.task_type == "nli":
            id2label = {0:"entailment", 1:"contradiction", 2:"neutral"}
        results = []
        y_true = []
        y_pred = []

        for batch in tqdm(test_loader):

            logits, probs, predicts, labels, _ = self.model.predict(batch)
            
            if data_set in ["val","test"]:
                y_true += list(labels)
                y_pred += list(predicts)

            for idx, (predict,prob,logit) in enumerate(zip(predicts,probs,logits)):
                pred = {
                            # "id": int(batch["id"][idx]),
                            'label': id2label[predict],
                            'sentence1': batch['texta'][idx],
                            'sentence2': batch['textb'][idx],
                            # "question": batch['question'][idx],
                            # "answer": batch['choice'][idx][self.label_classes_reverse[predict]],
                            # "choice": batch['choice'][idx],
                            "probs": prob.tolist(),
                            # "logits": logit.tolist(),
                        }
                
                if data_set=="unlabeled":
                    if prob[predict] > threshold:
                        pred["unlabeled_set"] = batch['unlabeled_set'][idx]
                        results.append(pred)
                else:
                    results.append(pred)

        return results, y_true, y_pred


class TextGenerateEvaluator(object):
    def __init__(self, args, model, data_model, save_path):
        super().__init__()
        self.args, self.model, self.data_model, self.save_path = args, model, data_model, save_path
        self.task_type = args.task_type
        

    def save_to_file(self, data_set, results, y_true, y_pred):
        
        if data_set in ["val","test"]:
            with open(self.save_path+f"{data_set}_set_predictions.json", 'w') as f:
                for result in results:
                    result = json.dumps(result,ensure_ascii=False)
                    f.write(result+"\n")

        elif data_set=="train":
            with open(self.save_path+"labeled_set_predictions.json", 'w') as f:
                for result in results:
                    result = json.dumps(result,ensure_ascii=False)
                    f.write(result+"\n")

        logger.info("Evaluation file saved at {}".format(self.save_path))


    def evaluation(self, mode, data_set):
        self.data_model.setup(mode)
        
        if data_set=="val":
            test_loader = self.data_model.val_dataloader()
        elif data_set=="test":
            test_loader = self.data_model.test_dataloader()
        elif data_set=="train":
            test_loader = self.data_model.train_dataloader()
            
        results, y_true, y_pred = self.inference(test_loader=test_loader, data_set=data_set)
        self.save_to_file(data_set, results, y_true, y_pred)

        TP, total_pred, total_true = 0, 0, 0
        for i in range(len(y_true)):
            true_list = y_true[i].split('|')
            pred_list = y_pred[i].split('|')
            pred_list = list(set(pred_list))

            for pred in pred_list:
                if pred in true_list:
                    TP += 1

            total_pred += len(pred_list)
            total_true += len(true_list)

        precision = TP / total_pred
        recall = TP / total_true
        f1 = float(2 * (precision * recall)) / (precision + recall) if precision + recall > 0 else 0.0
        
        logger.info("precision: {}".format(precision))
        logger.info("recall: {}".format(recall))
        logger.info("f1: {}".format(f1))
        return f1

    def inference(self, test_loader, data_set):
        results = []
        y_true = []
        y_pred = []

        for batch in tqdm(test_loader):

            _, _, predicts, labels = self.model.predict(batch)
            
            if data_set in ["val","test"]:
                y_true += list(labels)
                y_pred += list(predicts)

            for idx, predict in enumerate(predicts):
            
                pred = {
                        "id": int(batch["id"][idx]),
                        'label': predict
                    }

                results.append(pred)
    
        return results, y_true, y_pred


        