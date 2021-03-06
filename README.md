# panoptic_coco_scipt

New folder train_data_annotated and  val_data_annotated ,they have your annotated_image and json from  software labelme. 

install :
https://github.com/cocodataset/panopticapi

https://github.com/wkentaro/labelme


COCO2017 panoptic struction is:
COCO---(folder)
-----annotations(folder)
----------panoptic_train2017(folder)
----------panoptic_val2017(folder)
----------instances_train2017.json (file)
----------instances_val2017.json (file)
----------panoptic_train2017.json (file)
----------panoptic_val2017.json (file)
-----images(folder)
----------test2017(folder)
----------train2017(folder)
----------val2017(folder)

run :
python labelme2panoptic_trian.py
python labelme2panoptic_val.py

Note:
  在生成image的时候由于labelme_to_coco 使用了 colorIndexMap，就是一对数组 对应 一组颜色映射，这个Map是固定的。我直接用image/palette.png 提取出来的这个颜色Map。
