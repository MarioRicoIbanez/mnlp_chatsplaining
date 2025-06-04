# DATA COLLECTION

This folder contains all the scripts necessary to load datasets of mcqa and open answer of STEM related content. There are two base scripts in which data is processed:


- `base_processor.py`: class to structure mcqa datasets.
- `base_openans_processor.py`: class to structure the open answer datasets.

# DATASETS

Then we have the different individual scripts which load given datasets from huggingface and other sources and processes and saves them. The data currently processed is:

## SciQ

https://huggingface.co/datasets/allenai/sciq

## M1 open answer data from MNLP students

https://drive.google.com/file/d/1wvqJfzLfLBUGTJiF_Wo9EvucE9D8HL8H/view?usp=sharing


## GENERAL KNOWLEDGE

https://huggingface.co/datasets/pszemraj/unified-mcqa-all

600k

## MEDREASON: Medicine and biology

https://huggingface.co/datasets/UCSC-VLAA/MedReason

32k

## NVIDIA/OPENCODEREASON2: Coding

https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2

1.42M

## NVIDIA/OPENMATH: Mathematics

https://huggingface.co/datasets/nvidia/OpenMathReasoning

3.2M

## CHEMISTRY: Chemistry

https://huggingface.co/datasets/mlfoundations-dev/camel_chemistry_seed_science_20K

20k

## Engineering

https://huggingface.co/datasets/mlfoundations-dev/stackexchange_engineering

36k


# STRUCTURE

MCQA:

- question
- choices
- answer_index
- answer_text
- source
- explanation

Open answer:

- question
- answer
- source


