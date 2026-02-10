# RCoto: Training ASR with Wav2Vec2
# Last updated: 20250629 1624

# python3 train-wav2vec2-lm-hpc.py /dartfs/rc/lab/R/RCoto/asr-transfer-test/input-csv/wav2vec2-run1-train.csv /dartfs/rc/lab/R/RCoto/asr-transfer-test/input-csv/wav2vec2-run1-valid.csv /dartfs/rc/lab/R/RCoto/asr-transfer-test/input-csv/wav2vec2-run1-test.csv /dartfs/rc/lab/R/RCoto/asr-transfer-test/wav /dartfs/rc/lab/R/RCoto/asr-transfer-test/logs /dartfs/rc/lab/R/RCoto/asr-transfer-test/models "01" "fromScratch" 28 dz-fromScratch-noAugmentation-251mins-01

print(">>>> I started the training")

#============================================================================
# User inputs
# conditions: {transfer, fromScratch}
#============================================================================

import sys

asrLang = "baima"

csvTrain = sys.argv[1]
csvValid = sys.argv[2]
csvTest = sys.argv[3]
filenameCorrectKenlmModel= sys.argv[4]
folderAudioFiles = sys.argv[5]
folderLogFiles = sys.argv[6]
folderModelFiles = sys.argv[7]
runId = sys.argv[8]
condition = sys.argv[9]
desiredTrainEpochs = sys.argv[10]
saveTotalLimit = sys.argv[11]
perDeviceTrainBatchSize = sys.argv[12]
outputPrefix = sys.argv[13]
transferModelPath = ""

if ("transfer" in condition): transferModelPath = sys.argv[11]

print("csvTrain: " + csvTrain)
print("csvValid: " + csvValid)
print("csvTest: " + csvTest)
print("filenameCorrectKenlmModel: " + filenameCorrectKenlmModel)
print("folderAudioFiles: " + folderAudioFiles)
print("folderLogFiles: " + folderLogFiles)
print("folderModelFiles: " + folderModelFiles)
print("runId: " + runId)
print("condition: " + condition)
print("desiredTrainEpochs: " + desiredTrainEpochs)
print("saveTotalLimit: " + saveTotalLimit)
print("perDeviceTrainBatchSize: " + perDeviceTrainBatchSize)
if ("transfer" in condition): print("transferModelPath: " + transferModelPath) 

#============================================================================
# Load packages
#============================================================================

from datetime import datetime
currentDateAndTime = datetime.now()
startTime = str(currentDateAndTime)

import os

import transformers
import datasets
import torch

import numpy
import numpy as np

import pandas
import pandas as pd

from datasets import load_dataset, load_metric
from datasets import Dataset
from datasets import ClassLabel
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import TrainingArguments
from transformers import Trainer
from transformers import Wav2Vec2ProcessorWithLM

import pyctcdecode
from pyctcdecode import build_ctcdecoder

import random
import re
import json

import torchaudio
import librosa

from jiwer import wer
import statistics

print(">>>> I have finished loading the packages")

#============================================================================
# Load CSV files and prepare dataset
#============================================================================

dataTrain = pd.read_csv(csvTrain)
dataValid = pd.read_csv(csvValid)
dataTest = pd.read_csv(csvTest)

dataTrain.head()
dataValid.head()
dataTest.head()

common_voice_train = Dataset.from_pandas(dataTrain)
common_voice_test = Dataset.from_pandas(dataTest)
common_voice_valid = Dataset.from_pandas(dataValid)

common_voice_test_transcription = Dataset.from_pandas(dataTest)
common_voice_valid_transcription = Dataset.from_pandas(dataValid)

print(">>>> I finished loading the CSV")

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch
	
common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)
common_voice_valid = common_voice_valid.map(remove_special_characters)
	
def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}
  
vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)
vocab_valid = common_voice_valid.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_valid.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
vocab_dict["<s>"] = len(vocab_dict)
vocab_dict["</s>"] = len(vocab_dict)
len(vocab_dict)

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
	
print(">>>> I finished cleaning up the CSVs")

#============================================================================
# Create tokenizer
#============================================================================

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", bos_token="<s>", eos_token="</s>")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

processor.save_pretrained(folderModelFiles)

print("\n>>>> I finished creating the tokenizer\n")

#============================================================================
# Preprocess data
#============================================================================

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch

common_voice_train = common_voice_train.map(speech_file_to_array_fn, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(speech_file_to_array_fn, remove_columns=common_voice_test.column_names)
common_voice_valid = common_voice_valid.map(speech_file_to_array_fn, remove_columns=common_voice_valid.column_names)

def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, batch_size=8, num_proc=4, batched=True)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=8, num_proc=4, batched=True)
common_voice_valid = common_voice_valid.map(prepare_dataset, remove_columns=common_voice_valid.column_names, batch_size=8, num_proc=4, batched=True)

print(">>>> I finished processing the audio")

#============================================================================
# Prepare training
#============================================================================

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

print("\n>>>> I created the dataClass\n")

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

print("\n>>>> I created the DataCollator\n")

wer_metric = load_metric("wer")

print("\n>>>> I set the metric to WER\n")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

print("\n>>>> I created the computeMetrics\n")

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

print("\n>>>> I created the fromPretrained\n")

model.freeze_feature_extractor()

print("\n>>>> I created the feature extractor\n")

training_args = TrainingArguments(
  output_dir=folderModelFiles,
  group_by_length=True,
  per_device_train_batch_size=int(perDeviceTrainBatchSize),
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=int(desiredTrainEpochs),
  fp16=True,
  save_steps=400,
  eval_steps=100,
  logging_steps=50,
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=int(saveTotalLimit)
)

print("\n>>>> I created the Training Arguments\n")

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_valid,
    tokenizer=processor.feature_extractor
)

print("\n>>>> I created the Trainer\n")

#============================================================================
# Train
#============================================================================

print("I made to the point before the trainer.train()")
trainer.train()

#============================================================================
# Prepare stats
#============================================================================

# Calculate Levenshtein Distance between two strings (character distance)
# https://colab.research.google.com/github/Alexjmsherman/nlp_practicum_cohort3_instructor/blob/master/lessons/lesson_8_text_similarity/text_similarity_solution.ipynb#scrollTo=sSj3zYpq-sc1

def levenshtein(seq1, seq2):
    # create a matrix
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))

    # set col numbers (0, n-1)
    for x in range(size_x):
        matrix [x, 0] = x

    # set row numbers (0, n-1)
    for y in range(size_y):
        matrix [0, y] = y

    # calculate distance
    for x in range(1, size_x):
        for y in range(1, size_y):
            # if characters match do not increase distance
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = matrix[x-1, y-1]
            # if characters don't match increase min distance by 1
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )

    return (matrix[size_x - 1, size_y - 1])
    
sub_checkpoints = [name for name in os.listdir(folderModelFiles) if os.path.isdir(os.path.join(folderModelFiles, name))]
checkpoints = []
for f in sub_checkpoints:
  if ("checkpoint" in f and "ipynb" not in f):
    checkpoints.append(os.path.join(folderModelFiles, f))
print(checkpoints)

checkpointNums = []
for ch in checkpoints:
  checkpointNums.append(int(ch.split("-")[-1]))
checkpointNums.sort()
checkpoints.sort()

print(checkpointNums)
print(checkpoints)

#===============================================================================
# Evaluate checkpoints; calculate their word/character error rates and
# get the predictions for the sentences in the test set.
#===============================================================================

from multiprocessing import get_context

medianStats = ""
modes = ["withoutLM", "withLM"]

for ch in checkpointNums:

	checkpointNum = ch

	model = Wav2Vec2ForCTC.from_pretrained(folderModelFiles + "/checkpoint-"+str(ch)).to("cuda")
	processor = Wav2Vec2Processor.from_pretrained(folderModelFiles)

	vocab_dict = processor.tokenizer.get_vocab()
	sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

	tempDecoder = build_ctcdecoder(
		labels=list(sorted_vocab_dict.keys()),
		kenlm_model_path=filenameCorrectKenlmModel,
	)

	processor_with_lm = Wav2Vec2ProcessorWithLM(
		feature_extractor=processor.feature_extractor,
		tokenizer=processor.tokenizer,
		decoder=tempDecoder
	)

	prediction = []
	predictionLM = []
	reference = []
	paths = []

	print("Checkpoint: " + str(ch) + "\n")

	for i in range(0,len(common_voice_test)):

		# Prediction without LM

		input_dict = processor(common_voice_test[i]["input_values"], sampling_rate=16_000, return_tensors="pt", padding=True)
		logits = model(input_dict.input_values.to("cuda")).logits
		pred_ids = torch.argmax(logits, dim=-1)[0]

		#print("Prediction:")
		tempPrediction = processor.decode(pred_ids)
		prediction.append(tempPrediction)
		if (i == 0): print("Prediction:    " + str(tempPrediction))

		# Prediction with LM

		input_dict = processor_with_lm(common_voice_test[i]["input_values"], sampling_rate=16_000, return_tensors="pt", padding=True)
		logits = model(input_dict.input_values.to("cuda")).logits
		pred_ids = torch.argmax(logits, dim=-1)[0]

		#print("Prediction:")
		with get_context("fork").Pool(processes=4) as pool:
			predictionlm = processor_with_lm.batch_decode(logits.cpu().detach().numpy(), pool).text[0]
		predictionLM.append(predictionlm)
		if (i == 0): print("LM Prediction: " + str(predictionlm))

		#print("\nReference:")
		tempReference = common_voice_test_transcription[i]["sentence"].lower()
		reference.append(tempReference)
		if (i == 0): print("Reference:     " + str(tempReference) + "\n")

		path = common_voice_test_transcription[i]["path"]
		path = path.split("/")
		path = path[-1]
		paths.append(path)

	#cerList = []
	#werList = []
	#cerListLM = []
	#werListLM = []

	output = ""
	outputWithoutLM = ""
	outputWithLM = ""

	for m in modes:

		filename = "wav2vec2-"+str(condition)+"-res-" + str(runId) + "-ch" + str(ch) + "-" + m + ".csv"
		idThisRun = "wav2vec2-" + str(condition) + "-" + str(runId) + "-" + m

		output = "wav,src,res,loss,charDist,charLen,wordDist,wordLen,cer,wer,origin,condition,id,typeMonoTri,ngram\n"
		cerList = []
		werList = []

		modePredictions = prediction
		if (m=="withLM"): modePredictions = predictionLM

		for i in range(0,len(reference)):

			#print(str(i))

			levDistChar = levenshtein(reference[i],modePredictions[i])
			cer = levDistChar / len(reference[i])

			werSent = wer(reference[i],modePredictions[i])
			charLen = len(reference[i])
			charDist = levDistChar
			wordLen = len(modePredictions[i].split(' '))
			wordDist = werSent*wordLen

			cerList.append(cer)
			werList.append(werSent)

			wavFile = paths[i].replace(".wav","")

			output += wavFile + "," + reference[i] + "," + modePredictions[i] + ",," + str(charDist) + "," + str(charLen) + "," + str(wordDist) + "," + str(wordLen) + "," + str(round(cer,2)) + "," + str(round(werSent,2)) + "," + "wav2vec2-" + condition + "," + m + "-" + str(ch) + "," + str(idThisRun) + "," + "na" + "," + "na" + "\n"

		output = output[:-1]
		#print(output)

		cerMedian = statistics.median(cerList)
		werMedian = statistics.median(werList)

		medianStats += runId + "/" + str(ch) + " Median CER "+m+":\t" + str(round(cerMedian,3)) + "\n"
		medianStats += runId + "/" + str(ch) + " Median WER "+m+":\t" + str(round(werMedian,3)) + "\n\n"

		#print(runId + "/" + ch + " Median CER:\t" + str(round(cerMedian,3)))
		#print(runId + "/" + ch + " Median WER:\t" + str(round(werMedian,3)))

		f = open(folderLogFiles + "/" + filename, "w")
		f.write(output)
		f.close()

	#---------------

print(medianStats)

currentDateAndTime = datetime.now()
endTime = str(currentDateAndTime)

medianStats = "Run: " + runId + "\n\n" + medianStats + "\n\n" + "Start: " + startTime + "\nEnd: " + endTime

statsFilename = outputPrefix + "-stats-median.txt"

f = open(folderLogFiles + "/" + statsFilename, "w")
f.write(medianStats)
f.close()
