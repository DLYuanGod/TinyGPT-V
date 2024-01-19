import re    
import json
output_js=json.load(open("/mnt/share/zhengqing/zx/TinyGPT-V/['refcoco', 'refcoco+', 'refcocog']_testA.json"))
outputs=output_js['COCO_train2014_000000577725_3']

for output in outputs:
    try:
        integers = re.findall(r'\d+', output)
        pred_bbox = [int(num) for num in integers]
    except:
        continue    