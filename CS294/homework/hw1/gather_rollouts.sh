#!/bin/bash
NUM=100

for model in $(ls experts); do
	env_name="${model%.*}"
	python run_expert.py experts/$env_name.pkl $env_name --render --num_rollouts $NUM
done