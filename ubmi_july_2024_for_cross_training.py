# -*- coding: utf-8 -*-

"""
Requirements

!pip install transformers 
!pip install ray  
!pip install ray[tune]
!pip install datasets
!pip install sklearn
!pip install hpbandster ConfigSpace


"""

import numpy as np
import torch
import os, time
import json, random
from datasets import load_dataset
import datasets
from transformers import (
                    AutoConfig,
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                    Trainer,
                    TrainingArguments
)
from torch.utils.data import DataLoader
from sklearn.metrics import (
                          f1_score,
                          precision_recall_fscore_support,
                          accuracy_score,
                          balanced_accuracy_score,
                          precision_score,
                          recall_score,
)
import ray
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
#from ray.tune import CLIReporter

import gc

import argparse 

parser = argparse.ArgumentParser(description='specify the model name, the number of trials to run with bohb,the directory to store the checkpoints and the directory to store the results')
parser.add_argument('-model', type=str, required=True, help='e.g. /lustre/projects/cardiff/bert/bert-base-uncased')
parser.add_argument('-ntrials', type=int, default=30, help="number of trials with BOHB and Raytune")
parser.add_argument('-output', default="UBMI_cross_Results",type=str, help='Jsons with results on the validation and test sets will be saved here. HPO summary too.')
parser.add_argument('-chkpt_dir', default="./UBMI_cross_Checkpoints_",type=str, help='Checkpoints unsed by ray-tune and hf-transformers will be saved here')
parser.add_argument('--baseline', default=False, help="run the experiments on source concepts only", action='store_true')
parser.add_argument('-ray_tmp', default="/tmp/ray_tmp/",type=str, help="Set the path of the VERY large ray tmp files.")
parser.add_argument('-input_format', default="tagged_expr",type=str, help="input sentence preprocessing",choices=['expr', 'plain', 'tagged_expr',"tagged_pair","dependency"])
#parser.add_argument('-nsets', default=200,type=int, help="for debugging on less datasets")
               
args = parser.parse_args()

MODEL_NAME = args.model
N_TRIALS= args.ntrials
CHKPTS_PATH = args.chkpt_dir
OUTPUT_DIR = args.output
baseline = args.baseline
ray_tmp = args.ray_tmp
input_format=args.input_format


if OUTPUT_DIR=="UBMI_cross_Results":
    OUTPUT_DIR="UBMI_cross_Results__"+MODEL_NAME+"_"+input_format



#if CHKPTS_PATH=="UBMI_Checkpoints":
CHKPTS_PATH=CHKPTS_PATH+"__"+MODEL_NAME+"_"+input_format

    	
print("model name : ", MODEL_NAME)
print("number of BOHB trials : ", N_TRIALS)
print("Checkpoints directory : ", CHKPTS_PATH)
print("Results directory :", OUTPUT_DIR)
print("Baseline : ", baseline)

OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), OUTPUT_DIR))
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR,0o755)


tuning_summaries = os.path.join(OUTPUT_DIR,"TuningSummaries")
if not os.path.exists(tuning_summaries):
    os.mkdir(tuning_summaries,0o755)


evaldir = os.path.join(OUTPUT_DIR,"Evaluations")
if not os.path.exists(evaldir):
    os.mkdir(evaldir,0o755)
    
    


def set_reproducibillity(seed=None):
    if seed is None:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False



set_reproducibillity()

############# Load and tokenize datasets #####################################
# Data preprocessing functions


def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True,max_length=128) 



def add_expr_boundaries(context, position):
    cl = list(context)
    cl.insert(position[0],'<expr> '),cl.insert(position[1]+1,' </expr>')
    return ''.join(cl)


def add_target_and_expr_boundaries(context,ps,pt):
    cl = list(context)
    if ps[1] <= pt[0]:
        cl.insert(ps[0],'<expr> '),cl.insert(ps[1]+1,' </expr>')
        cl.insert(pt[0]+2,'<target> '),cl.insert(pt[1]+3,' </target>')
    elif pt[1] <= p[0]:
        cl.insert(pt[0],'<target> '),cl.insert(pt[1]+1,' </target>')
        cl.insert(ps[0]+2,'<expr> '),cl.insert(ps[1]+3,' </expr>')  
    return "".join(cl)


def add_pair_boundaries_and_dep(context,ps,pt,dep):
    cl = list(context)
    if ps[1] <= pt[0]:
        cl.insert(ps[0],'<expr>'),cl.insert(ps[1]+1,'</expr>')
        cl.insert(pt[0]+2,'<target>'),cl.insert(pt[1]+3,'</target>')
    elif pt[1] <= p[0]:
        cl.insert(pt[0],'<target>'),cl.insert(pt[1]+1,'</target>')
        cl.insert(ps[0]+2,'<expr>'),cl.insert(ps[1]+3,'</expr>')  
    cl="".join(cl)+"[SEP] "+dep
    return cl



def preprocess_context(ptype,e):
    if ptype=="plain":
         return e["context"]
    elif ptype=="expr":
        return e["expression"]
    elif ptype=="tagged_expr":
        return add_expr_boundaries(e["context"], e["position"])
    elif ptype=="tagged_pair":
        return add_target_and_expr_boundaries(e["context"], e["position"],e["target_position"])
    elif ptype=="dependency":
        return add_target_and_expr_boundaries(e["context"], e["position"],e["target_position"],e["dependency"])
    else:
        print("Unknown input format!")


# # Convert to transformers classification expected format


def Convert_to_long_context(data):
    dsplit = ["train","validation","test"]
    for dsplit in data:
        for i,x in enumerate(data[dsplit]):
            if len(x["previous_sentence"])>0:
                data[dsplit][i]["context"]=x["previous_sentence"]+" "+x["context"]
                data[dsplit][i]["position"][0]+=len(x["previous_sentence"])+1
                data[dsplit][i]["position"][1]+=len(x["previous_sentence"])+1
            if len(x["next_sentence"])>0:
                data[dsplit][i]["context"]=data[dsplit][i]["context"]+ " " + x["next_sentence"]
    return data



_DLABELS = {"m":1,"l":0, "rp":0}

def Convert(dataset, prepro = "tagged_expr" , pair=False, context="base", splits = ["train","validation","test"]):    
    #for i,s in enumerate(splits):
    if context=="3-sent":
        dataset = Convert_to_long_context(dataset) 
    #    for x in dataset[s]:
    dataset = dataset.map(lambda x: {"label":_DLABELS[x["label"]] })
    dataset = dataset.map(lambda x: {"text":preprocess_context(prepro,x) })
    if len(dataset["validation"])==0:
        dataset["validation"]=dataset["test"]
    return dataset





########################################################################################


def ray_hp_space(trial):
    return {
        "per_device_train_batch_size": tune.choice([4,8,16]),
        #"weight_decay": tune.uniform(0.0, 0.3),
        "num_train_epochs": tune.choice([1,2,4,6,10,12]),
        "seed":tune.choice([1, 2, 3]),#, 3]),
        "learning_rate": tune.uniform(1e-6, 5e-5),
    }


def model_init(n):
    model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,num_labels=n, return_dict=True)
    model.resize_token_embeddings(len(tokenizer))
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    macrof1 = f1_score(labels,predictions,average="macro")
    return {"eval_macro_f1":macrof1}


def EvaluationMetrics(logits,y_true):
    y_pred = np.argmax(logits, axis=-1)
    analysis={}
    analysis["predictions"]=[int(x*1) for x in y_pred]
    analysis["accuracy"]={}
    #micro
    analysis["micro-f1"]= round(f1_score(y_true,y_pred,average="micro"),4)
    analysis["micro-precision"]=round(precision_score(y_true,y_pred,average="micro"),4)
    analysis["micro-recall"]=round(recall_score(y_true,y_pred, average="micro"),4)
    # macro
    analysis["macro-f1"]=round(f1_score(y_true,y_pred,average="macro"),4)
    analysis["macro-precision"]=round(precision_score(y_true,y_pred,average="macro"),4)
    analysis["macro-recall"]=round(recall_score(y_true,y_pred,average="macro"),4)
    #weighted
    analysis["weighted-f1"]=round(f1_score(y_true,y_pred,average="weighted"),4)
    analysis["weighted-precision"]=round(precision_score(y_true,y_pred,average="weighted"),4)
    analysis["weighted-recall"]=round(recall_score(y_true,y_pred,average="weighted"),4)
    #Accuracy
    analysis["accuracy"]["balanced"]=round(balanced_accuracy_score(y_true,y_pred),4)
    analysis["accuracy"]["standard"]=round(accuracy_score(y_true,y_pred),4)
    analysis["class_prf1"]=[list(x) for x in precision_recall_fscore_support(y_true,y_pred)]
    for i,x in enumerate(analysis["class_prf1"][:-1]):
        analysis["class_prf1"][i] = [round(y,4) for y in x]
    analysis["class_prf1"][-1]= [int(x*1) for x in analysis["class_prf1"][-1 ]]
    return analysis 


cl_datasets =[
	"JC",
	"GUT_RANDSPLIT",
	"MOH_REG_LEMSPLIT",
	"TSVET_AN_FILTERED_RANDSPLIT",
	"PVC_RANDSPLIT",
	"MAD_FEWSHOTS_RANDSPLIT",
	"MAGPIE_FILTERED_RANDSPLIT",
	"VUAC_MW_RANDSPLIT",
	"TONG",
	"NEWSMET"
	]

listsets = json.load(open("the_10_sets.json"))

hf_cl_sets = []
for x in listsets:
	dataloaders = datasets.DatasetDict({
		"train": datasets.Dataset.from_list(x[:800]),
		"validation": datasets.Dataset.from_list(x[800:900]),
		"test": datasets.Dataset.from_list(x[900:])
		})
	hf_cl_sets.append(dataloaders)



### Preprocess data ############################

## load the tokenizer and add special tokens to it

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
special_tokens_dict = {'additional_special_tokens': ['<expr>','</expr>','<target>','</target>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)



## Preprocess according to the different options passed to Convert 

# ## Preprocessing types
# 
# - plain
# - tagged_expr
# - tagged_pair
# - dependency
# 

all_data, results = {},{}

for i,d in enumerate(hf_cl_sets):
    try:
        all_data[cl_datasets[i]]=Convert(d, prepro = "tagged_expr", context="base").map(tokenize_function, batched=True)
    except:
        print("not loaded :" , cl_datasets[i])
        
print("Correctly preprocessed sets :",len(all_data))



######################### TUNE HP for all the models  ####################################

def tune_func(config):
    tune.util.wait_for_gpu()
    train()



def ConvertGeneralTrialSummaryToJsonSerialisable(d):
    #rerun_best.run_summary.trial_dataframes
    summary = {}
    for p in d:
        summary[p]=d[p].to_dict()
    return summary


def HPO(dataset_name,ray_hp_space_function,keep_chkpt=1,ntrials=50,ray_tmp_path="/mnt/storage_joanne/ray_tmp",outputdir=CHKPTS_PATH):
    #Initialize ray and set the large tmp directort path
    ray.init(_temp_dir=ray_tmp_path,num_gpus=1)
    set_reproducibillity()
    # Training Arguments
    training_args = TrainingArguments(
              disable_tqdm=True,
              per_device_eval_batch_size=8,
              evaluation_strategy="epoch",
              save_strategy="epoch",
              output_dir = outputdir,
              full_determinism=True,
              data_seed=42
    )
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="eval_macro_f1",
        mode="max",
        #max_t=100,
    )
    algo = TuneBOHB( 
        metric="eval_macro_f1",
        mode="max",
        seed=42
    )
    # Huggingface trainer
    label_set = set([e['label'] for e in all_data[dataset_name]["train"]])
    trainer = Trainer(
        args=training_args,
        train_dataset=all_data[dataset_name]["train"],
        eval_dataset=all_data[dataset_name]["validation"],
        model_init=lambda: model_init(len(label_set)),
        compute_metrics=compute_metrics,
    )
    # Tune HP
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        search_alg=algo,
        scheduler=scheduler,
        hp_space=ray_hp_space_function,
        n_trials=ntrials,
        keep_checkpoints_num=keep_chkpt,
        local_dir=outputdir,
        # resources_per_trial={"cpu":1,"gpu":1},
        name=dataset_name,
        #log_to_file=True,
    )
    ray.shutdown()
    return best_trial

###############################################################################


for dataset_name in all_data: #
    # Perform HPO here, saving only 1 checkpoint
    best_trial = HPO(dataset_name,ray_hp_space,keep_chkpt=1,ntrials=N_TRIALS,ray_tmp_path=ray_tmp)
    
    # Find the Best Model info  
    trials = list(best_trial.run_summary.trial_dataframes.keys())
    trials_info = []
    for t in trials:
    	bestmt = best_trial.run_summary.trial_dataframes[t].sort_values(by="objective",ascending=False).iloc[0]
    	trials_info.append([t,bestmt["trial_id"],bestmt["objective"],bestmt["epoch"]])   
    trials_info.sort(key=lambda w: w[2], reverse = True)
    hp = best_trial.run_summary.get_all_configs()[trials_info[0][0]]
    hp["best_epoch"]=trials_info[0][-1]
    trials_details = ConvertGeneralTrialSummaryToJsonSerialisable(best_trial.run_summary.trial_dataframes)  
    too_many_models = best_trial.run_summary._get_trial_paths()
    # Write a dummy HPO space function for reproductibility of the best results based on the objective
    def fake_hp_space_for_best_model(trial):
        return {
            "per_device_train_batch_size": tune.choice([hp["per_device_train_batch_size"]]),
            "num_train_epochs": tune.choice([hp["num_train_epochs"]]),
            "seed":tune.choice([hp["seed"]]),
            "learning_rate":tune.choice([hp["learning_rate"]])
    } 
     
    # Rerun the best model and save all the checkpoints
    del best_trial
    rerun_best = HPO(dataset_name,fake_hp_space_for_best_model,keep_chkpt=hp["num_train_epochs"],ntrials=1, ray_tmp_path=ray_tmp,outputdir="WINNWERS_"+dataset_name)
    
    # Get all the info related to the best checkpoint
    trial_id = rerun_best.run_summary.get_best_trial("eval_macro_f1","max")
    win_id = rerun_best.run_id
    win_name = rerun_best.run_summary.get_best_trial("eval_macro_f1","max")
    chkpt = rerun_best.run_summary.get_best_checkpoint(trial_id,"eval_macro_f1",mode="max")
    winner_chkpt_path = chkpt._local_path
    chkpt_name = winner_chkpt_path.split("/")[-1] 
    winner_trial_path = "/".join(winner_chkpt_path.split("/")[:-1])  
    #rerun_best.run_summary._get_trial_paths()[0]
    
    # Write all info in result json file
    results[dataset_name] = {"trials_info":{},"test":{},"validation":{}}
    results[dataset_name]["hyperparameters"] = hp
    results[dataset_name]["model"] = MODEL_NAME
    rerun_details = ConvertGeneralTrialSummaryToJsonSerialisable(rerun_best.run_summary.trial_dataframes)
    rerun_objective = rerun_details[winner_trial_path]["objective"][hp["best_epoch"]-1]
    results[dataset_name]["trials_info"]["id"] = str(win_id)
    results[dataset_name]["trials_info"]["objective"] = [trials_info[0][2],rerun_objective]
    
    #Check that we have exact reproduction of the first run with the second run
    assert results[dataset_name]["trials_info"]["objective"][0]==results[dataset_name]["trials_info"]["objective"][1], print(results[dataset_name]["trials_info"]["objective"])
    
    results[dataset_name]["trials_info"]["name"] = str(trial_id)
    results[dataset_name]["trials_info"]["best_chkpt_path"]= winner_chkpt_path
    results[dataset_name]["trials_info"]["best_trial_path"]= winner_trial_path
    
    # Save all trial results separately here in case we need to come back to it later
    with open(tuning_summaries+"/tuning_all_info_"+dataset_name+".json", "w") as outfile:
        json.dump(trials_details,outfile,indent="\t")   
    
    del rerun_best
    gc.collect()
    # Load best model from checkpoint
    # Strange path issue :
    content = os.listdir(winner_chkpt_path)
    c = [x for x in content if x[:10]=="checkpoint"]
    if not "config.json" in content:
        model = AutoModelForSequenceClassification.from_pretrained(winner_chkpt_path+"/"+c[0])
    else:
        model = AutoModelForSequenceClassification.from_pretrained(win_chkpt_path)  
        
    # Predictions for the and all metrics for the validation set and the test set   
    test_trainer = Trainer(
       model,
       args = TrainingArguments(
           output_dir=CHKPTS_PATH,
           per_device_eval_batch_size=8
           )
    )       
    
    # Testing the Best Model on the 10 val and test sets
    results[dataset_name]["ten_sets"]={}
 
    for dn in cl_datasets:
    	results[dataset_name]["ten_sets"][dn]={"test":{},"validation":{}}
    	test_pred = test_trainer.predict(all_data[dn]["test"])
    	val_pred = test_trainer.predict(all_data[dn]["validation"])
    	results[dataset_name]["ten_sets"][dn]["test"]["metrics"] = EvaluationMetrics(test_pred[0],test_pred[1])
    	results[dataset_name]["ten_sets"][dn]["validation"]["metrics"] =EvaluationMetrics(val_pred[0],val_pred[1])  
    
    # Save all results for one dataset separetely in case the training is interrupted after only a part of the datasets are processed
    with open(evaldir+"/results_"+dataset_name+".json", "w") as outfile:
        json.dump(results[dataset_name],outfile,indent="\t")
        
    # for storage explosion due to raytune optimisation             
    os.system("rm -r "+CHKPTS_PATH)
                 
# Save final dictionary on all the datasets   
with open(OUTPUT_DIR+"/results_all_sets.json", "w") as outfile:
    json.dump(results,outfile,indent="\t")



