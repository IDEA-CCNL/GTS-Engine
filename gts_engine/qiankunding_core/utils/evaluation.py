import json
from time import clock_settime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

    print("global_acc:", global_acc)

    return eval_results

class Evaluator(object):
    # def __init__(self, args, model, data_model, save_path, mode, data_set):
    #     super().__init__()
    #     self.args, self.model, self.data_model, self.save_path, self.mode, self.data_set = args, model, data_model, save_path, mode, data_set
    
    def __init__(self, args, model, data_model, save_path):
        super().__init__()
        self.args, self.model, self.data_model, self.save_path = args, model, data_model, save_path

        self.task_type = args.task_type
        self.label_classes = self.data_model.label_classes
        print("label_classes",self.label_classes)
        with open(self.save_path+"label_classes.json", 'w') as f:
                result = json.dumps(self.label_classes,ensure_ascii=False)
                f.write(result+"\n")
        self.label_classes_reverse = {v:k for k,v in self.label_classes.items()}
        print("label_classes_reverse",self.label_classes_reverse)


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
            # print(y_true, y_pred)
            # print(eval_results)
            with open(self.save_path+f"{data_set}_set_confusion_matrix.json", "w") as f:
                json.dump(eval_results, f, indent=4, ensure_ascii=False)

        elif data_set=="train":
            with open(self.save_path+"labeled_set_predictions.json", 'w') as f:
                for result in results:
                    result = json.dumps(result,ensure_ascii=False)
                    f.write(result+"\n")


        print("Evaluation file saved at {}".format(self.save_path))

    def inference(self,test_loader,data_set):
        results = []
        y_true = []
        y_pred = []

        for batch in tqdm(test_loader):

            logits, probs, predicts, labels, _ = self.model.predict(batch)
            # print(predicts,labels)
            if data_set in ["val","test"]:
                y_true += list(labels)
                y_pred += list(predicts)

        
            for idx, (predict,prob,logit) in enumerate(zip(predicts,probs,logits)):
                
                pred = {
                # "id": int(batch["id"][idx]),
                'text': batch['sentence'][idx],
                'label': self.label_classes_reverse[predict],
                # 'content': batch['sentence'][idx],
                "probs": prob.tolist(),
            }

                results.append(pred)
        return results, y_true, y_pred

    def evaluation(self, mode, data_set):
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

        results, y_true, y_pred = self.inference(test_loader=test_loader,data_set=data_set)
        self.save_to_file(data_set, results, y_true, y_pred)

class SentencePairEvaluator(Evaluator):
    def inference(self,test_loader,data_set):
        if self.task_type == "similarity":
            id2label = {0:0, 1:1}
        elif self.task_type == "nli":
            id2label = {0:"entailment", 1:"contradiction", 2:"neutral"}
        results = []
        y_true = []
        y_pred = []

        for batch in tqdm(test_loader):

            logits, probs, predicts, labels, _ = self.model.predict(batch)
            # print(predicts,labels)
            if data_set in ["val","test"]:
                y_true += list(labels)
                y_pred += list(predicts)

        
            for idx, (predict,prob,logit) in enumerate(zip(predicts,probs,logits)):
                pred = {
                            "id": int(batch["id"][idx]),
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
                    if prob[predict] > self.prob_threshold:
                        pred["unlabeled_set"] = batch['unlabeled_set'][idx]
                        results.append(pred)
                else:
                    results.append(pred)

        return results, y_true, y_pred

def evaluation(args, model, data_model, save_path, mode, data_set):
    data_model.setup(mode)
    tokenizer = data_model.tokenizer
    label_classes = data_model.label_classes
    # print("model.label_classes:",model.label_classes)
    print(label_classes)
    label_classes_reverse = {v:k for k,v in label_classes.items()}
    print(label_classes_reverse)
    if data_set=="test":
        test_loader = data_model.test_dataloader()
    
    results = []


    y_true = []
    y_pred = []

    # special_tuning_methods = ["fine_tuning","pet","pet_joint_finetune","mlm_tuning","label_guided_tuning_cls","label_guided_tuning","UnifiedMC","UGC_tuning","T5_tuning","cocolm_pet"]
    # special_tuning_methods += ["text_match","token_classification"]

    for batch in tqdm(test_loader):

        logits, probs, predicts, labels, _ = model.predict(batch)
        # print(predicts,labels)
        if data_set=="test":
            y_true += list(labels)
            y_pred += list(predicts)
        
    
        for idx, (predict,prob,logit) in enumerate(zip(predicts,probs,logits)):
            
            pred = {
                # "id": int(batch["id"][idx]),
                'text': batch['sentence'][idx],
                'label': label_classes_reverse[predict],
                # 'content': batch['sentence'][idx],
                "probs": prob.tolist(),
            }

                # print({'content': batch['sentence'][idx],'label': label_classes[predict]})

            results.append(pred)


    
    if data_set=="test":
        with open(save_path+"test_set_predictions.json", 'w') as f:
            for result in results:
                result = json.dumps(result,ensure_ascii=False)
                f.write(result+"\n")

        
            eval_results = result_eval(y_true, y_pred, label_names=label_classes)
            # print(y_true, y_pred)
            # print(eval_results)
            with open(save_path+"test_set_confusion_matrix.json", "w") as f:
                json.dump(eval_results, f, indent=4, ensure_ascii=False)



    with open(save_path+"label_classes.json", 'w') as f:
            result = json.dumps(label_classes,ensure_ascii=False)
            f.write(result+"\n")

    print("Evaluation file saved at {}".format(save_path))