# Joint Model across Multiple Tasks
This is code of model for Joint Model across Multiple Tasks  [SimpleTOD](https://proceedings.neurips.cc/paper/2020/hash/e946209592563be0f01c844ab2170f0c-Abstract.html).
It corresponds to model in article *MMConv: an Environment for Multimodal Conversational Search across Multiple Domains*, section 5.4. To reproduce results shown in paper, follow the steps below:

### input generation
```python generate_inputs.py```
The preprocessed inputs will be stored in folder /resources. Or you may directly run the blocks in generate_inputs.ipynb.

### model training
```sh train_multitask.sh $CUDA_VISIBLE_DEVICES $MODEL $MODEL_NAME $BATCH```

$MODEL is the name of model group used as backbone, and $MODEL_NAME is the name of specific model or the path to that model pre-downloaded.

One runnable example is like this:
```sh train_multitask.sh 0,1,2,3 gpt2 gpt2 4```

### model evaluation
```python eval_simpletod.py $MODEL $BATCH $checkpoint```
Here $checkpoint is one of model folder saved in ./checkpoints.

Or simply run eval_simpletod.ipynb and assign checkpoint number in the script.
