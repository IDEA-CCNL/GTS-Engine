from torch import frac
from teacher_core.dataloaders.text_classification.dataloader import TaskDataset, TaskDataModel
from teacher_core.dataloaders.text_classification.dataloader_UnifiedMC import TaskDataModelUnifiedMC


from teacher_core.models.text_classification.bert_UnifiedMC import BertUnifiedMC


tuning_methods_config = {
    #文本分类
    "UnifiedMC":{"DataModel":TaskDataModelUnifiedMC,"TuningModel":BertUnifiedMC},
}
