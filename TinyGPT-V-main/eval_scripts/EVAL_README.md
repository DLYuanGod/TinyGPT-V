## Evaluation Instruction for TinyGPT-V

### Data preparation
Images download
Image source | Download path
--- | :---:
gqa | <a href="https://drive.google.com/drive/folders/1-dF-cgFwstutS4qq2D9CFQTDS0UTmIft?usp=drive_link">annotations</a> &nbsp;&nbsp;  <a href="https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip">images</a> 
hateful meme |  <a href="https://github.com/faizanahemad/facebook-hateful-memes">images and annotations</a> 
iconqa |  <a href="https://iconqa.github.io/#download">images and annotation</a>
vizwiz |  <a href="https://vizwiz.org/tasks-and-datasets/vqa/">images and annotation</a>

### Evaluation dataset structure

```
${MINIGPTv2_EVALUATION_DATASET}
├── gqa
│   └── test_balanced_questions.json
│   ├── testdev_balanced_questions.json
│   ├── gqa_images
├── hateful_meme
│   └── hm_images
│   ├── dev.jsonl
├── iconvqa
│   └── iconvqa_images
│   ├── choose_text_val.json
├── vizwiz
│   └── vizwiz_images
│   ├── val.json
├── vsr
│   └── vsr_images
...
```



### config file setup

Set **llama_model** to the path of Phi model.  
Set **ckpt** to the path of our pretrained model.  
Set **eval_file_path** to the path of the annotation files for each evaluation data.  
Set **img_path** to the img_path for each evaluation dataset.  
Set **save_path** to the save_path for each evaluation dataset.    

in [eval_configs/minigptv2_benchmark_evaluation.yaml](../eval_configs/benchmark_evaluation.yaml) 





### start evaluating visual question answering

port=port_number  
cfg_path=/path/to/eval_configs/benchmark_evaluation.yaml 

dataset names:  
| vizwiz | iconvqa | gqa | vsr | hm |
| ------- | -------- | -------- |-------- | -------- |


```
torchrun --master-port ${port} --nproc_per_node 1 eval_vqa.py \
 --cfg-path ${cfg_path} --dataset vizwiz,iconvqa,gqa,vsr,hm
```




