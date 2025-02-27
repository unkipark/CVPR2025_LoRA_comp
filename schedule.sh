# pueue add -- python classification_lora.py mode=test model=tic_promptmodel_first2 quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/cls_1.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_hp quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar)

# pueue add -- python classification_lora.py mode=test model=tic_promptmodel_first2 quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/cls_2.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_hp quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/base_codec_2.pth.tar)

# pueue add -- python classification_lora.py mode=test model=tic_promptmodel_first2 quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/cls_3.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_hp quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/base_codec_3.pth.tar)

# pueue add -- python classification_lora.py mode=test model=tic_promptmodel_first2 quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/cls_4.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_hp quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/base_codec_4.pth.tar)

# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath cls/pre/1/2024-11-04_09-04-34checkpoint_best_loss.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/base_codec_2.pth.tar) checkpoint=$(realpath cls/pre/2/2024-11-04_09-04-37checkpoint_best_loss.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/base_codec_3.pth.tar) checkpoint=$(realpath cls/pre/3/2024-11-04_09-04-41checkpoint_best_loss.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/base_codec_4.pth.tar) checkpoint=$(realpath cls/pre/4/2024-11-04_09-04-44checkpoint_best_loss.pth.tar)

# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath hp/1/checkpoint_best_loss.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/base_codec_2.pth.tar) checkpoint=$(realpath hp/2/checkpoint_best_loss.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/base_codec_3.pth.tar) checkpoint=$(realpath hp/3/checkpoint_best_loss.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/base_codec_4.pth.tar) checkpoint=$(realpath hp/4/checkpoint_best_loss.pth.tar)

# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath od/1/checkpoint_best_loss.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath is/1/checkpoint_best_loss.pth.tar)

# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/base_codec_2.pth.tar) checkpoint=$(realpath od/2/checkpoint_best_loss.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/base_codec_2.pth.tar) checkpoint=$(realpath is/2/checkpoint_best_loss.pth.tar)

# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/base_codec_3.pth.tar) checkpoint=$(realpath od/3/checkpoint_best_loss.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/base_codec_3.pth.tar) checkpoint=$(realpath is/3/checkpoint_best_loss.pth.tar)

# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/base_codec_4.pth.tar) checkpoint=$(realpath od/4/checkpoint_best_loss.pth.tar)
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/base_codec_4.pth.tar) checkpoint=$(realpath is/4/checkpoint_best_loss.pth.tar)

# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath hp/1/checkpoint_best_loss.pth.tar) num_images=200
# pueue add -- python classification_lora.py mode=TTT model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath hp/1/checkpoint_best_loss.pth.tar) num_images=200

# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/base_codec_2.pth.tar) checkpoint=$(realpath hp/2/checkpoint_best_loss.pth.tar) num_images=200
# pueue add -- python classification_lora.py mode=TTT model=tic_svdlora quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/base_codec_2.pth.tar) checkpoint=$(realpath hp/2/checkpoint_best_loss.pth.tar) num_images=200

# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/base_codec_3.pth.tar) checkpoint=$(realpath hp/3/checkpoint_best_loss.pth.tar) num_images=200
# pueue add -- python classification_lora.py mode=TTT model=tic_svdlora quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/base_codec_3.pth.tar) checkpoint=$(realpath hp/3/checkpoint_best_loss.pth.tar) num_images=200

# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/base_codec_4.pth.tar) checkpoint=$(realpath hp/4/checkpoint_best_loss.pth.tar) num_images=200
# pueue add -- python classification_lora.py mode=TTT model=tic_svdlora quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/base_codec_4.pth.tar) checkpoint=$(realpath hp/4/checkpoint_best_loss.pth.tar) num_images=200

# pueue add -- python segmentation_lora.py mode=test model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath hp/1/checkpoint_best_loss.pth.tar) num_images=20
# pueue add -- python segmentation_lora.py mode=TTT model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath hp/1/checkpoint_best_loss.pth.tar) num_images=20 train_image_form=1
# pueue add -- python segmentation_lora.py mode=TTT model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath hp/1/checkpoint_best_loss.pth.tar) num_images=20 train_image_form=0

# pueue add -- python segmentation_lora.py mode=test model=tic_svdlora quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/base_codec_2.pth.tar) checkpoint=$(realpath hp/2/checkpoint_best_loss.pth.tar) num_images=20
# pueue add -- python segmentation_lora.py mode=TTT model=tic_svdlora quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/base_codec_2.pth.tar) checkpoint=$(realpath hp/2/checkpoint_best_loss.pth.tar) num_images=20 train_image_form=1
# pueue add -- python segmentation_lora.py mode=TTT model=tic_svdlora quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/base_codec_2.pth.tar) checkpoint=$(realpath hp/2/checkpoint_best_loss.pth.tar) num_images=20 train_image_form=0

# pueue add -- python segmentation_lora.py mode=test model=tic_svdlora quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/base_codec_3.pth.tar) checkpoint=$(realpath hp/3/checkpoint_best_loss.pth.tar) num_images=20
# pueue add -- python segmentation_lora.py mode=TTT model=tic_svdlora quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/base_codec_3.pth.tar) checkpoint=$(realpath hp/3/checkpoint_best_loss.pth.tar) num_images=20 train_image_form=1
# pueue add -- python segmentation_lora.py mode=TTT model=tic_svdlora quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/base_codec_3.pth.tar) checkpoint=$(realpath hp/3/checkpoint_best_loss.pth.tar) num_images=20 train_image_form=0

# pueue add -- python segmentation_lora.py mode=test model=tic_svdlora quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/base_codec_4.pth.tar) checkpoint=$(realpath hp/4/checkpoint_best_loss.pth.tar) num_images=20
# pueue add -- python segmentation_lora.py mode=TTT model=tic_svdlora quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/base_codec_4.pth.tar) checkpoint=$(realpath hp/4/checkpoint_best_loss.pth.tar) num_images=20 train_image_form=1
# pueue add -- python segmentation_lora.py mode=TTT model=tic_svdlora quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/base_codec_4.pth.tar) checkpoint=$(realpath hp/4/checkpoint_best_loss.pth.tar) num_images=20 train_image_form=0

# pueue add -- python classification_lora.py mode=test model=tic_promptmodel_first2 quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/cls_1.pth.tar) num_images=200
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath hp/1/checkpoint_best_loss.pth.tar) num_images=200
# pueue add -- python classification_lora.py mode=TTT model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath hp/1/checkpoint_best_loss.pth.tar) num_images=200

# pueue add -- python classification_lora.py mode=test model=tic_promptmodel_first2 quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/cls_2.pth.tar) num_images=200
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/base_codec_2.pth.tar) checkpoint=$(realpath hp/2/checkpoint_best_loss.pth.tar) num_images=200
# pueue add -- python classification_lora.py mode=TTT model=tic_svdlora quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/base_codec_2.pth.tar) checkpoint=$(realpath hp/2/checkpoint_best_loss.pth.tar) num_images=200

# pueue add -- python classification_lora.py mode=test model=tic_promptmodel_first2 quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/cls_3.pth.tar) num_images=200
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/base_codec_3.pth.tar) checkpoint=$(realpath hp/3/checkpoint_best_loss.pth.tar) num_images=200
# pueue add -- python classification_lora.py mode=TTT model=tic_svdlora quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/base_codec_3.pth.tar) checkpoint=$(realpath hp/3/checkpoint_best_loss.pth.tar) num_images=200

# pueue add -- python classification_lora.py mode=test model=tic_promptmodel_first2 quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/cls_4.pth.tar) num_images=200
# pueue add -- python classification_lora.py mode=test model=tic_svdlora quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/base_codec_4.pth.tar) checkpoint=$(realpath hp/4/checkpoint_best_loss.pth.tar) num_images=200
# pueue add -- python classification_lora.py mode=TTT model=tic_svdlora quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/base_codec_4.pth.tar) checkpoint=$(realpath hp/4/checkpoint_best_loss.pth.tar) num_images=200

# pueue add -- python classification_lora_head.py port=1234 quality_level=1 num_images=20
# pueue add -- python classification_lora_node.py port=1234

# pueue add -g cu0 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=0 task_model=point_rend quality_level=1 checkpoint_task=cls
# pueue add -g cu1 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=1 task_model=point_rend quality_level=2 checkpoint_task=cls
# pueue add -g cu2 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=2 task_model=point_rend quality_level=3 checkpoint_task=cls
# pueue add -g cu3 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=3 task_model=point_rend quality_level=4 checkpoint_task=cls

# pueue add -g cu0 -- python detection_lora.py mode=test model=tic_hp gpu_id=0 task_model=point_rend quality_level=1 checkpoint_task=base_codec
# pueue add -g cu1 -- python detection_lora.py mode=test model=tic_hp gpu_id=1 task_model=point_rend quality_level=2 checkpoint_task=base_codec
# pueue add -g cu2 -- python detection_lora.py mode=test model=tic_hp gpu_id=2 task_model=point_rend quality_level=3 checkpoint_task=base_codec
# pueue add -g cu3 -- python detection_lora.py mode=test model=tic_hp gpu_id=3 task_model=point_rend quality_level=4 checkpoint_task=base_codec

# pueue add -g cu0 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=0 task_model=point_rend quality_level=1 checkpoint_task=cls
# pueue add -g cu1 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=1 task_model=point_rend quality_level=2 checkpoint_task=cls
# pueue add -g cu2 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=2 task_model=point_rend quality_level=3 checkpoint_task=cls
# pueue add -g cu3 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=3 task_model=point_rend quality_level=4 checkpoint_task=cls

# pueue add -g cu0 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=0 task_model=faster_rcnn quality_level=1 checkpoint_task=det
# pueue add -g cu1 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=1 task_model=faster_rcnn quality_level=2 checkpoint_task=det
# pueue add -g cu2 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=2 task_model=faster_rcnn quality_level=3 checkpoint_task=det
# pueue add -g cu3 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=3 task_model=faster_rcnn quality_level=4 checkpoint_task=det

# pueue add -g cu0 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=0 task_model=mask_rcnn quality_level=1 checkpoint_task=seg
# pueue add -g cu1 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=1 task_model=mask_rcnn quality_level=2 checkpoint_task=seg
# pueue add -g cu2 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=2 task_model=mask_rcnn quality_level=3 checkpoint_task=seg
# pueue add -g cu3 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=3 task_model=mask_rcnn quality_level=4 checkpoint_task=seg


# pueue add -g cu0 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=0 task_model=faster_rcnn quality_level=1 checkpoint_task=cls
# pueue add -g cu1 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=1 task_model=faster_rcnn quality_level=2 checkpoint_task=cls
# pueue add -g cu2 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=2 task_model=faster_rcnn quality_level=3 checkpoint_task=cls
# pueue add -g cu3 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=3 task_model=faster_rcnn quality_level=4 checkpoint_task=cls

# pueue add -g cu0 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=0 task_model=mask_rcnn quality_level=1 checkpoint_task=cls
# pueue add -g cu1 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=1 task_model=mask_rcnn quality_level=2 checkpoint_task=cls
# pueue add -g cu2 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=2 task_model=mask_rcnn quality_level=3 checkpoint_task=cls
# pueue add -g cu3 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=3 task_model=mask_rcnn quality_level=4 checkpoint_task=cls


# pueue add -g cu0 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=0 task_model=point_rend quality_level=1 checkpoint_task=cls
# pueue add -g cu1 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=1 task_model=point_rend quality_level=2 checkpoint_task=cls
# pueue add -g cu2 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=2 task_model=point_rend quality_level=3 checkpoint_task=cls
# pueue add -g cu3 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=3 task_model=point_rend quality_level=4 checkpoint_task=cls

# pueue add -g cu0 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=0 task_model=detr quality_level=1 checkpoint_task=cls
# pueue add -g cu1 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=1 task_model=detr quality_level=2 checkpoint_task=cls
# pueue add -g cu2 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=2 task_model=detr quality_level=3 checkpoint_task=cls
# pueue add -g cu3 -- python detection_lora.py mode=test model=tic_promptmodel_first2 gpu_id=3 task_model=detr quality_level=4 checkpoint_task=cls

# pueue add -g cu0 -- python detection_lora.py mode=test model=tic_hp gpu_id=0 task_model=point_rend quality_level=1 checkpoint_task=base_codec
# pueue add -g cu1 -- python detection_lora.py mode=test model=tic_hp gpu_id=1 task_model=point_rend quality_level=2 checkpoint_task=base_codec
# pueue add -g cu2 -- python detection_lora.py mode=test model=tic_hp gpu_id=2 task_model=point_rend quality_level=3 checkpoint_task=base_codec
# pueue add -g cu3 -- python detection_lora.py mode=test model=tic_hp gpu_id=3 task_model=point_rend quality_level=4 checkpoint_task=base_codec

# pueue add -g cu0 -- python detection_lora.py mode=test model=tic_hp gpu_id=0 task_model=detr quality_level=1 checkpoint_task=base_codec
# pueue add -g cu1 -- python detection_lora.py mode=test model=tic_hp gpu_id=1 task_model=detr quality_level=2 checkpoint_task=base_codec
# pueue add -g cu2 -- python detection_lora.py mode=test model=tic_hp gpu_id=2 task_model=detr quality_level=3 checkpoint_task=base_codec
# pueue add -g cu3 -- python detection_lora.py mode=test model=tic_hp gpu_id=3 task_model=detr quality_level=4 checkpoint_task=base_codec

# pueue add -g cu0 -- python detection_lora.py mode=TTT model=tic_svdlora gpu_id=0 task_model=point_rend quality_level=1 checkpoint_task=base_codec checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar)
# pueue add -g cu1 -- python detection_lora.py mode=TTT model=tic_svdlora gpu_id=1 task_model=point_rend quality_level=2 checkpoint_task=base_codec checkpoint=$(realpath cls/2/checkpoint_best_loss.pth.tar)
# pueue add -g cu2 -- python detection_lora.py mode=TTT model=tic_svdlora gpu_id=2 task_model=point_rend quality_level=3 checkpoint_task=base_codec checkpoint=$(realpath cls/3/checkpoint_best_loss.pth.tar)
# pueue add -g cu3 -- python detection_lora.py mode=TTT model=tic_svdlora gpu_id=3 task_model=point_rend quality_level=4 checkpoint_task=base_codec checkpoint=$(realpath cls/4/checkpoint_best_loss.pth.tar)

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=test model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=mask_rcnn num_images=10
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=TTT  model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=mask_rcnn num_images=10

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=test model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=faster_rcnn num_images=10
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=TTT  model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=faster_rcnn num_images=10

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=test model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=point_rend num_images=100
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=TTT  model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=point_rend num_images=100

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=TTT  model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=point_rend num_images=10 VPT_lmbda=2.0
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=TTT  model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=point_rend num_images=10 VPT_lmbda=4.0

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=TTT  model=tic_svdlora quality_level=1 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=point_rend num_images=100 VPT_lmbdas=[8,4,2,0.8]
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=TTT  model=tic_svdlora quality_level=2 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath cls/2/checkpoint_best_loss.pth.tar) task_model=point_rend num_images=100 VPT_lmbdas=[8,4,2,0.8]
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=TTT  model=tic_svdlora quality_level=3 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath cls/3/checkpoint_best_loss.pth.tar) task_model=point_rend num_images=100 VPT_lmbdas=[8,4,2,0.8]
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=TTT  model=tic_svdlora quality_level=4 checkpoint_pre_trained=$(realpath examples/utils/base_codec_1.pth.tar) checkpoint=$(realpath cls/4/checkpoint_best_loss.pth.tar) task_model=point_rend num_images=100 VPT_lmbdas=[8,4,2,0.8]

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=test model=tic_promptmodel_first2 quality_level=1 checkpoint_task=base_codec task_model=point_rend num_images=100 # 132
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=test model=tic_promptmodel_first2 quality_level=2 checkpoint_task=base_codec task_model=point_rend num_images=100
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=test model=tic_promptmodel_first2 quality_level=3 checkpoint_task=base_codec task_model=point_rend num_images=100
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=test model=tic_promptmodel_first2 quality_level=4 checkpoint_task=base_codec task_model=point_rend num_images=100

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=test model=tic_promptmodel_first2 quality_level=1 checkpoint_task=cls task_model=point_rend num_images=100 # 136
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=test model=tic_promptmodel_first2 quality_level=2 checkpoint_task=cls task_model=point_rend num_images=100
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=test model=tic_promptmodel_first2 quality_level=3 checkpoint_task=cls task_model=point_rend num_images=100
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=test model=tic_promptmodel_first2 quality_level=4 checkpoint_task=cls task_model=point_rend num_images=100

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 task_model=point_rend mode=TTT model=tic_svdlora quality_level=1 checkpoint_task=base_codec checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) VPT_lmbdas=[8,4,2,0.8]
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 task_model=point_rend mode=TTT model=tic_svdlora quality_level=2 checkpoint_task=base_codec checkpoint=$(realpath cls/2/checkpoint_best_loss.pth.tar) VPT_lmbdas=[8,4,2,0.8]
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 task_model=point_rend mode=TTT model=tic_svdlora quality_level=3 checkpoint_task=base_codec checkpoint=$(realpath cls/3/checkpoint_best_loss.pth.tar) VPT_lmbdas=[8,4,2,0.8]
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 task_model=point_rend mode=TTT model=tic_svdlora quality_level=4 checkpoint_task=base_codec checkpoint=$(realpath cls/4/checkpoint_best_loss.pth.tar) VPT_lmbdas=[8,4,2,0.8]

# python detection_lora.py gpu_id=3 task_model=point_rend mode=TTT model=tic_svdlora quality_level=4 checkpoint_task=base_codec checkpoint=$(realpath cls/4/checkpoint_best_loss.pth.tar) VPT_lmbdas=[8,4,2,0.8]


# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=test model=tic_promptmodel_first2 quality_level=1 checkpoint_task=det task_model=detr
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=test model=tic_promptmodel_first2 quality_level=2 checkpoint_task=det task_model=detr
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=test model=tic_promptmodel_first2 quality_level=3 checkpoint_task=det task_model=detr
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=test model=tic_promptmodel_first2 quality_level=4 checkpoint_task=det task_model=detr

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=test model=tic_promptmodel_first2 quality_level=1 checkpoint_task=seg task_model=point_rend
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=test model=tic_promptmodel_first2 quality_level=2 checkpoint_task=seg task_model=point_rend
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=test model=tic_promptmodel_first2 quality_level=3 checkpoint_task=seg task_model=point_rend
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=test model=tic_promptmodel_first2 quality_level=4 checkpoint_task=seg task_model=point_rend

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=test model=tic_promptmodel_first2 quality_level=1 checkpoint_task=cls task_model=detr
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=test model=tic_promptmodel_first2 quality_level=2 checkpoint_task=cls task_model=detr
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=test model=tic_promptmodel_first2 quality_level=3 checkpoint_task=cls task_model=detr
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=test model=tic_promptmodel_first2 quality_level=4 checkpoint_task=cls task_model=detr

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=test model=tic_promptmodel_first2 quality_level=1 checkpoint_task=cls task_model=point_rend
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=test model=tic_promptmodel_first2 quality_level=2 checkpoint_task=cls task_model=point_rend
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=test model=tic_promptmodel_first2 quality_level=3 checkpoint_task=cls task_model=point_rend
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=test model=tic_promptmodel_first2 quality_level=4 checkpoint_task=cls task_model=point_rend

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=test model=tic_svdlora quality_level=1 checkpoint_task=base_codec checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=point_rend
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=test model=tic_svdlora quality_level=2 checkpoint_task=base_codec checkpoint=$(realpath cls/2/checkpoint_best_loss.pth.tar) task_model=point_rend
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=test model=tic_svdlora quality_level=3 checkpoint_task=base_codec checkpoint=$(realpath cls/3/checkpoint_best_loss.pth.tar) task_model=point_rend
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=test model=tic_svdlora quality_level=4 checkpoint_task=base_codec checkpoint=$(realpath cls/4/checkpoint_best_loss.pth.tar) task_model=point_rend

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=test model=tic_svdlora quality_level=1 checkpoint_task=base_codec checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=detr
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=test model=tic_svdlora quality_level=2 checkpoint_task=base_codec checkpoint=$(realpath cls/2/checkpoint_best_loss.pth.tar) task_model=detr
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=test model=tic_svdlora quality_level=3 checkpoint_task=base_codec checkpoint=$(realpath cls/3/checkpoint_best_loss.pth.tar) task_model=detr
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=test model=tic_svdlora quality_level=4 checkpoint_task=base_codec checkpoint=$(realpath cls/4/checkpoint_best_loss.pth.tar) task_model=detr


# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=TTT model=tic_svdlora quality_level=1 checkpoint_task=base_codec checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=detr
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=TTT model=tic_svdlora quality_level=2 checkpoint_task=base_codec checkpoint=$(realpath cls/2/checkpoint_best_loss.pth.tar) task_model=detr
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=TTT model=tic_svdlora quality_level=3 checkpoint_task=base_codec checkpoint=$(realpath cls/3/checkpoint_best_loss.pth.tar) task_model=detr
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=TTT model=tic_svdlora quality_level=4 checkpoint_task=base_codec checkpoint=$(realpath cls/4/checkpoint_best_loss.pth.tar) task_model=detr

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=TTT model=tic_svdlora quality_level=1 checkpoint_task=base_codec checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=point_rend
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=TTT model=tic_svdlora quality_level=2 checkpoint_task=base_codec checkpoint=$(realpath cls/2/checkpoint_best_loss.pth.tar) task_model=point_rend
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=TTT model=tic_svdlora quality_level=3 checkpoint_task=base_codec checkpoint=$(realpath cls/3/checkpoint_best_loss.pth.tar) task_model=point_rend
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=TTT model=tic_svdlora quality_level=4 checkpoint_task=base_codec checkpoint=$(realpath cls/4/checkpoint_best_loss.pth.tar) task_model=point_rend

# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=test model=tic_promptmodel_first2 quality_level=1 checkpoint_task=det task_model=faster_rcnn num_images=500
# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=TTT model=tic_svdlora quality_level=1 checkpoint_task=base_codec checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=faster_rcnn learning_rate=1e-2 num_images=500
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=TTT model=tic_svdlora quality_level=1 checkpoint_task=base_codec checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=faster_rcnn num_images=500

# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=test model=tic_promptmodel_first2 quality_level=1 checkpoint_task=det task_model=faster_rcnn num_images=1000
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=TTT model=tic_svdlora quality_level=1 checkpoint_task=base_codec checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=faster_rcnn learning_rate=1e-2 num_images=1000
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=TTT model=tic_svdlora quality_level=1 checkpoint_task=base_codec checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=faster_rcnn num_images=1000

# pueue add -- python classification_lora_head.py port=1231 mode=test model=tic_promptmodel_first2 quality_level=1 checkpoint_task=cls checkpoint=null
# pueue add -- python classification_lora_head.py port=1232 mode=test model=tic_promptmodel_first2 quality_level=2 checkpoint_task=cls checkpoint=null
# pueue add -- python classification_lora_head.py port=1241 mode=TTT model=tic_svdlora quality_level=1 checkpoint_task=base_codec
# pueue add -- python classification_lora_head.py port=1242 mode=TTT model=tic_svdlora quality_level=2 checkpoint_task=base_codec

# pueue add -- python classification_lora_head.py port=1111 mode=test model=tic_promptmodel_first2 quality_level=3 checkpoint_task=cls checkpoint=null
# pueue add -- python classification_lora_head.py port=1112 mode=TTT model=tic_svdlora quality_level=3 checkpoint_task=base_codec
# pueue add -- python classification_lora_head.py port=1113 mode=test model=tic_promptmodel_first2 quality_level=4 checkpoint_task=cls checkpoint=null
# pueue add -- python classification_lora_head.py port=1114 mode=TTT model=tic_svdlora quality_level=4 checkpoint_task=base_codec

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=TTT model=tic_svdlora quality_level=1 checkpoint_task=base_codec checkpoint=$(realpath cls/1/checkpoint_best_loss.pth.tar) task_model=faster_rcnn learning_rate=1e-2 num_images=1000
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=TTT model=tic_svdlora quality_level=2 checkpoint_task=base_codec checkpoint=$(realpath cls/2/checkpoint_best_loss.pth.tar) task_model=faster_rcnn learning_rate=1e-2 num_images=1000
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=TTT model=tic_svdlora quality_level=3 checkpoint_task=base_codec checkpoint=$(realpath cls/3/checkpoint_best_loss.pth.tar) task_model=faster_rcnn learning_rate=1e-2 num_images=1000
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=TTT model=tic_svdlora quality_level=4 checkpoint_task=base_codec checkpoint=$(realpath cls/4/checkpoint_best_loss.pth.tar) task_model=faster_rcnn learning_rate=1e-2 num_images=1000

# pueue add -g cu0 -- python detection_lora.py gpu_id=0 mode=test model=tic_promptmodel_first2 quality_level=1 checkpoint_task=det task_model=faster_rcnn num_images=1000
# pueue add -g cu1 -- python detection_lora.py gpu_id=1 mode=test model=tic_promptmodel_first2 quality_level=2 checkpoint_task=det task_model=faster_rcnn num_images=1000
# pueue add -g cu2 -- python detection_lora.py gpu_id=2 mode=test model=tic_promptmodel_first2 quality_level=3 checkpoint_task=det task_model=faster_rcnn num_images=1000
# pueue add -g cu3 -- python detection_lora.py gpu_id=3 mode=test model=tic_promptmodel_first2 quality_level=4 checkpoint_task=det task_model=faster_rcnn num_images=1000

# pueue add -g det -- python detection_lora_head.py port=4321 mode=test model=tic_promptmodel_first2 quality_level=1 task_model=faster_rcnn checkpoint_task=det checkpoint=null
# pueue add -g cu0 -- python detection_lora_node.py port=4321 gpu_id=0 num_images=[0,1250]
# pueue add -g cu1 -- python detection_lora_node.py port=4321 gpu_id=1 num_images=[1250,2500]
# pueue add -g cu2 -- python detection_lora_node.py port=4321 gpu_id=2 num_images=[2500,3750]
# pueue add -g cu3 -- python detection_lora_node.py port=4321 gpu_id=3 num_images=[3750,5000]

# pueue add -g det -- python detection_lora_head.py port=5321 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec
# pueue add -g cu0 -- python detection_lora_node.py port=5321 gpu_id=0 num_images=[0,1250]
# pueue add -g cu1 -- python detection_lora_node.py port=5321 gpu_id=1 num_images=[1250,2500]
# pueue add -g cu2 -- python detection_lora_node.py port=5321 gpu_id=2 num_images=[2500,3750]
# pueue add -g cu3 -- python detection_lora_node.py port=5321 gpu_id=3 num_images=[3750,5000]

# pueue add -g det -- python detection_lora_head.py port=5331 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=1e-2
# pueue add -g cu0 -- python detection_lora_node.py port=5331 gpu_id=0 num_images=[0,1250]
# pueue add -g cu1 -- python detection_lora_node.py port=5331 gpu_id=1 num_images=[1250,2500]
# pueue add -g cu2 -- python detection_lora_node.py port=5331 gpu_id=2 num_images=[2500,3750]
# pueue add -g cu3 -- python detection_lora_node.py port=5331 gpu_id=3 num_images=[3750,5000]

# pueue add -g det -- python detection_lora_head.py port=5331 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=1e-2 num_images=100 scheduler=cosine
# pueue add -g cu0 -- python detection_lora_node.py port=5331 gpu_id=0 num_images=[0,25]
# pueue add -g cu1 -- python detection_lora_node.py port=5331 gpu_id=1 num_images=[25,50]
# pueue add -g cu2 -- python detection_lora_node.py port=5331 gpu_id=2 num_images=[50,75]
# pueue add -g cu3 -- python detection_lora_node.py port=5331 gpu_id=3 num_images=[75,100]

# pueue add -g det -- python detection_lora_head.py port=5332 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=1e-2 num_images=100 scheduler=multistep
# pueue add -g cu0 -- python detection_lora_node.py port=5332 gpu_id=0 num_images=[4000,4025]
# pueue add -g cu1 -- python detection_lora_node.py port=5332 gpu_id=1 num_images=[4025,4050]
# pueue add -g cu2 -- python detection_lora_node.py port=5332 gpu_id=2 num_images=[4050,4075]
# pueue add -g cu3 -- python detection_lora_node.py port=5332 gpu_id=3 num_images=[4075,4100]

# pueue add -g det -- python detection_lora_head.py port=5333 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=1e-2 num_images=100 scheduler=cosine
# pueue add -g cu0 -- python detection_lora_node.py port=5333 gpu_id=0 num_images=[4000,4025]
# pueue add -g cu1 -- python detection_lora_node.py port=5333 gpu_id=1 num_images=[4025,4050]
# pueue add -g cu2 -- python detection_lora_node.py port=5333 gpu_id=2 num_images=[4050,4075]
# pueue add -g cu3 -- python detection_lora_node.py port=5333 gpu_id=3 num_images=[4075,4100]

# pueue add -g det -- python detection_lora_head.py port=5331 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=1e-2 num_images=100 scheduler=multistep
# pueue add -g cu0 -- python detection_lora_node.py port=5331 gpu_id=0 num_images=[0,25]
# pueue add -g cu1 -- python detection_lora_node.py port=5331 gpu_id=1 num_images=[25,50]
# pueue add -g cu2 -- python detection_lora_node.py port=5331 gpu_id=2 num_images=[50,75]
# pueue add -g cu3 -- python detection_lora_node.py port=5331 gpu_id=3 num_images=[75,100]

# pueue add -g det -- python detection_lora_head.py port=5332 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=9e-3 num_images=100 scheduler=cosine
# pueue add -g cu0 -- python detection_lora_node.py port=5332 gpu_id=0 num_images=[0,25]
# pueue add -g cu1 -- python detection_lora_node.py port=5332 gpu_id=1 num_images=[25,50]
# pueue add -g cu2 -- python detection_lora_node.py port=5332 gpu_id=2 num_images=[50,75]
# pueue add -g cu3 -- python detection_lora_node.py port=5332 gpu_id=3 num_images=[75,100]

# pueue add -g det -- python detection_lora_head.py port=5333 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=8e-3 num_images=100 scheduler=cosine
# pueue add -g cu0 -- python detection_lora_node.py port=5333 gpu_id=0 num_images=[0,25]
# pueue add -g cu1 -- python detection_lora_node.py port=5333 gpu_id=1 num_images=[25,50]
# pueue add -g cu2 -- python detection_lora_node.py port=5333 gpu_id=2 num_images=[50,75]
# pueue add -g cu3 -- python detection_lora_node.py port=5333 gpu_id=3 num_images=[75,100]

# pueue add -g det -- python detection_lora_head.py port=5334 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=7e-3 num_images=100 scheduler=cosine
# pueue add -g cu0 -- python detection_lora_node.py port=5334 gpu_id=0 num_images=[0,25]
# pueue add -g cu1 -- python detection_lora_node.py port=5334 gpu_id=1 num_images=[25,50]
# pueue add -g cu2 -- python detection_lora_node.py port=5334 gpu_id=2 num_images=[50,75]
# pueue add -g cu3 -- python detection_lora_node.py port=5334 gpu_id=3 num_images=[75,100]

# pueue add -g det -- python detection_lora_head.py port=5335 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=6e-3 num_images=100 scheduler=cosine
# pueue add -g cu0 -- python detection_lora_node.py port=5335 gpu_id=0 num_images=[0,25]
# pueue add -g cu1 -- python detection_lora_node.py port=5335 gpu_id=1 num_images=[25,50]
# pueue add -g cu2 -- python detection_lora_node.py port=5335 gpu_id=2 num_images=[50,75]
# pueue add -g cu3 -- python detection_lora_node.py port=5335 gpu_id=3 num_images=[75,100]

# pueue add -g det -- python detection_lora_head.py port=5336 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=5e-3 num_images=100 scheduler=cosine
# pueue add -g cu0 -- python detection_lora_node.py port=5336 gpu_id=0 num_images=[0,25]
# pueue add -g cu1 -- python detection_lora_node.py port=5336 gpu_id=1 num_images=[25,50]
# pueue add -g cu2 -- python detection_lora_node.py port=5336 gpu_id=2 num_images=[50,75]
# pueue add -g cu3 -- python detection_lora_node.py port=5336 gpu_id=3 num_images=[75,100]

# pueue add -g det -- python detection_lora_head.py port=5332 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=9e-3 num_images=100 scheduler=cosine
# pueue add -g cu0 -- python detection_lora_node.py port=5332 gpu_id=0 num_images=[4000,4025]
# pueue add -g cu1 -- python detection_lora_node.py port=5332 gpu_id=1 num_images=[4025,4050]
# pueue add -g cu2 -- python detection_lora_node.py port=5332 gpu_id=2 num_images=[4050,4075]
# pueue add -g cu3 -- python detection_lora_node.py port=5332 gpu_id=3 num_images=[4075,4100]

# pueue add -g det -- python detection_lora_head.py port=5333 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=8e-3 num_images=100 scheduler=cosine
# pueue add -g cu0 -- python detection_lora_node.py port=5333 gpu_id=0 num_images=[4000,4025]
# pueue add -g cu1 -- python detection_lora_node.py port=5333 gpu_id=1 num_images=[4025,4050]
# pueue add -g cu2 -- python detection_lora_node.py port=5333 gpu_id=2 num_images=[4050,4075]
# pueue add -g cu3 -- python detection_lora_node.py port=5333 gpu_id=3 num_images=[4075,4100]

# pueue add -g det -- python detection_lora_head.py port=5334 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=7e-3 num_images=100 scheduler=cosine
# pueue add -g cu0 -- python detection_lora_node.py port=5334 gpu_id=0 num_images=[4000,4025]
# pueue add -g cu1 -- python detection_lora_node.py port=5334 gpu_id=1 num_images=[4025,4050]
# pueue add -g cu2 -- python detection_lora_node.py port=5334 gpu_id=2 num_images=[4050,4075]
# pueue add -g cu3 -- python detection_lora_node.py port=5334 gpu_id=3 num_images=[4075,4100]

# pueue add -g det -- python detection_lora_head.py port=5335 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=6e-3 num_images=100 scheduler=cosine
# pueue add -g cu0 -- python detection_lora_node.py port=5335 gpu_id=0 num_images=[4000,4025]
# pueue add -g cu1 -- python detection_lora_node.py port=5335 gpu_id=1 num_images=[4025,4050]
# pueue add -g cu2 -- python detection_lora_node.py port=5335 gpu_id=2 num_images=[4050,4075]
# pueue add -g cu3 -- python detection_lora_node.py port=5335 gpu_id=3 num_images=[4075,4100]

# pueue add -g det -- python detection_lora_head.py port=5336 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn checkpoint_task=base_codec learning_rate=5e-3 num_images=100 scheduler=cosine
# pueue add -g cu0 -- python detection_lora_node.py port=5336 gpu_id=0 num_images=[4000,4025]
# pueue add -g cu1 -- python detection_lora_node.py port=5336 gpu_id=1 num_images=[4025,4050]
# pueue add -g cu2 -- python detection_lora_node.py port=5336 gpu_id=2 num_images=[4050,4075]
# pueue add -g cu3 -- python detection_lora_node.py port=5336 gpu_id=3 num_images=[4075,4100]

# pueue add -g mask_1e2 -- python detection_lora_head.py port=5331 mode=TTT model=tic_svdlora quality_level=1 task_model=mask_rcnn checkpoint_task=base_codec learning_rate=1e-2
# pueue add -g mask_1e2 -- python detection_lora_head.py port=5332 mode=TTT model=tic_svdlora quality_level=2 task_model=mask_rcnn checkpoint_task=base_codec learning_rate=1e-2
# pueue add -g mask_1e2 -- python detection_lora_head.py port=5333 mode=TTT model=tic_svdlora quality_level=3 task_model=mask_rcnn checkpoint_task=base_codec learning_rate=1e-2
# pueue add -g mask_1e2 -- python detection_lora_head.py port=5334 mode=TTT model=tic_svdlora quality_level=4 task_model=mask_rcnn checkpoint_task=base_codec learning_rate=1e-2

# pueue add -g etc -- python classification_lora.py mode=test model=tic_promptmodel_first2 gpu_id=0 quality_level=1 checkpoint_task=cls checkpoint=null
# pueue add -g etc -- python classification_lora.py mode=test model=tic_promptmodel_first2 gpu_id=1 quality_level=2 checkpoint_task=cls checkpoint=null
# pueue add -g etc -- python classification_lora.py mode=test model=tic_promptmodel_first2 gpu_id=2 quality_level=3 checkpoint_task=cls checkpoint=null
# pueue add -g etc -- python classification_lora.py mode=test model=tic_promptmodel_first2 gpu_id=3 quality_level=4 checkpoint_task=cls checkpoint=null

# pueue add -g etc -- python classification_lora_head.py port=8001 mode=test model=tic_svdlora quality_level=1 checkpoint_task=base_codec
# pueue add -g etc_cu0 -- python classification_lora_node.py port=8001 gpu_id=0 address=127.0.0.1
# pueue add -g etc_cu1 -- python classification_lora_node.py port=8001 gpu_id=1 address=127.0.0.1
# pueue add -g etc_cu2 -- python classification_lora_node.py port=8001 gpu_id=2 address=127.0.0.1
# pueue add -g etc_cu3 -- python classification_lora_node.py port=8001 gpu_id=3 address=127.0.0.1

# pueue add -g etc -- python classification_lora_head.py port=8002 mode=test model=tic_svdlora quality_level=2 checkpoint_task=base_codec
# pueue add -g etc_cu0 -- python classification_lora_node.py port=8002 gpu_id=0 address=127.0.0.1
# pueue add -g etc_cu1 -- python classification_lora_node.py port=8002 gpu_id=1 address=127.0.0.1
# pueue add -g etc_cu2 -- python classification_lora_node.py port=8002 gpu_id=2 address=127.0.0.1
# pueue add -g etc_cu3 -- python classification_lora_node.py port=8002 gpu_id=3 address=127.0.0.1

# pueue add -g etc -- python classification_lora_head.py port=8003 mode=test model=tic_svdlora quality_level=3 checkpoint_task=base_codec
# pueue add -g etc_cu0 -- python classification_lora_node.py port=8003 gpu_id=0 address=127.0.0.1
# pueue add -g etc_cu1 -- python classification_lora_node.py port=8003 gpu_id=1 address=127.0.0.1
# pueue add -g etc_cu2 -- python classification_lora_node.py port=8003 gpu_id=2 address=127.0.0.1
# pueue add -g etc_cu3 -- python classification_lora_node.py port=8003 gpu_id=3 address=127.0.0.1

# pueue add -g etc -- python classification_lora_head.py port=8004 mode=test model=tic_svdlora quality_level=4 checkpoint_task=base_codec
# pueue add -g etc_cu0 -- python classification_lora_node.py port=8004 gpu_id=0 address=127.0.0.1
# pueue add -g etc_cu1 -- python classification_lora_node.py port=8004 gpu_id=1 address=127.0.0.1
# pueue add -g etc_cu2 -- python classification_lora_node.py port=8004 gpu_id=2 address=127.0.0.1
# pueue add -g etc_cu3 -- python classification_lora_node.py port=8004 gpu_id=3 address=127.0.0.1

# pueue add -g od_close -- python detection_lora_head.py port=9001 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn transtic_weight=base_codec svd_pre_weight=od learning_rate=1e-2
# pueue add -g od_close -- python detection_lora_head.py port=9002 mode=TTT model=tic_svdlora quality_level=2 task_model=faster_rcnn transtic_weight=base_codec svd_pre_weight=od learning_rate=1e-2
# pueue add -g od_close -- python detection_lora_head.py port=9003 mode=TTT model=tic_svdlora quality_level=3 task_model=faster_rcnn transtic_weight=base_codec svd_pre_weight=od learning_rate=1e-2
# pueue add -g od_close -- python detection_lora_head.py port=9004 mode=TTT model=tic_svdlora quality_level=4 task_model=faster_rcnn transtic_weight=base_codec svd_pre_weight=od learning_rate=1e-2

# pueue add -g is-open -- python detection_lora_head.py port=1201 mode=TTT model=tic_svdlora quality_level=1 task_model=mask_rcnn transtic_weight=base_codec svd_pre_weight=is learning_rate=1e-2
# pueue add -g cu0 -- python detection_lora_node.py address=127.0.0.1 port=1201 gpu_id=0
# pueue add -g cu1 -- python detection_lora_node.py address=127.0.0.1 port=1201 gpu_id=1
# pueue add -g cu2 -- python detection_lora_node.py address=127.0.0.1 port=1201 gpu_id=2
# pueue add -g cu3 -- python detection_lora_node.py address=127.0.0.1 port=1201 gpu_id=3

# pueue add -g is-open -- python detection_lora_head.py port=1202 mode=TTT model=tic_svdlora quality_level=2 task_model=mask_rcnn transtic_weight=base_codec svd_pre_weight=is learning_rate=1e-2
# pueue add -g cu0 -- python detection_lora_node.py address=127.0.0.1 port=1202 gpu_id=0
# pueue add -g cu1 -- python detection_lora_node.py address=127.0.0.1 port=1202 gpu_id=1
# pueue add -g cu2 -- python detection_lora_node.py address=127.0.0.1 port=1202 gpu_id=2
# pueue add -g cu3 -- python detection_lora_node.py address=127.0.0.1 port=1202 gpu_id=3

# pueue add -g is-open -- python detection_lora_head.py port=1203 mode=TTT model=tic_svdlora quality_level=3 task_model=mask_rcnn transtic_weight=base_codec svd_pre_weight=is learning_rate=1e-2
# pueue add -g cu0 -- python detection_lora_node.py address=127.0.0.1 port=1203 gpu_id=0
# pueue add -g cu1 -- python detection_lora_node.py address=127.0.0.1 port=1203 gpu_id=1
# pueue add -g cu2 -- python detection_lora_node.py address=127.0.0.1 port=1203 gpu_id=2
# pueue add -g cu3 -- python detection_lora_node.py address=127.0.0.1 port=1203 gpu_id=3

# pueue add -g is-open -- python detection_lora_head.py port=1204 mode=TTT model=tic_svdlora quality_level=4 task_model=mask_rcnn transtic_weight=base_codec svd_pre_weight=is learning_rate=1e-2
# pueue add -g cu0 -- python detection_lora_node.py address=127.0.0.1 port=1204 gpu_id=0
# pueue add -g cu1 -- python detection_lora_node.py address=127.0.0.1 port=1204 gpu_id=1
# pueue add -g cu2 -- python detection_lora_node.py address=127.0.0.1 port=1204 gpu_id=2
# pueue add -g cu3 -- python detection_lora_node.py address=127.0.0.1 port=1204 gpu_id=3

# pueue add -g is-open -- python detection_lora_head.py port=7001 mode=TTT model=tic_svdlora quality_level=1 task_model=mask_rcnn transtic_weight=base_codec svd_pre_weight=is learning_rate=1e-2
# pueue add -g is-open -- python detection_lora_head.py port=7002 mode=TTT model=tic_svdlora quality_level=2 task_model=mask_rcnn transtic_weight=base_codec svd_pre_weight=is learning_rate=1e-2
# pueue add -g is-open -- python detection_lora_head.py port=7003 mode=TTT model=tic_svdlora quality_level=3 task_model=mask_rcnn transtic_weight=base_codec svd_pre_weight=is learning_rate=1e-2
# pueue add -g is-open -- python detection_lora_head.py port=7004 mode=TTT model=tic_svdlora quality_level=4 task_model=mask_rcnn transtic_weight=base_codec svd_pre_weight=is learning_rate=1e-2

pueue add -g left -l is-open-real -- python detection_lora_head.py port=6121 mode=TTT model=tic_svdlora quality_level=1 task_model=mask_rcnn transtic_weight=base_codec svd_pre_weight=cls learning_rate=1e-2
pueue add -g left -l is-open-real -- python detection_lora_head.py port=6122 mode=TTT model=tic_svdlora quality_level=2 task_model=mask_rcnn transtic_weight=base_codec svd_pre_weight=cls learning_rate=1e-2
pueue add -g left -l is-open-real -- python detection_lora_head.py port=6123 mode=TTT model=tic_svdlora quality_level=3 task_model=mask_rcnn transtic_weight=base_codec svd_pre_weight=cls learning_rate=1e-2
pueue add -g left -l is-open-real -- python detection_lora_head.py port=6124 mode=TTT model=tic_svdlora quality_level=4 task_model=mask_rcnn transtic_weight=base_codec svd_pre_weight=cls learning_rate=1e-2

pueue add -g left -l od-open -- python detection_lora_head.py port=5411 mode=TTT model=tic_svdlora quality_level=1 task_model=faster_rcnn transtic_weight=base_codec svd_pre_weight=cls learning_rate=1e-2
pueue add -g left -l od-open -- python detection_lora_head.py port=5412 mode=TTT model=tic_svdlora quality_level=2 task_model=faster_rcnn transtic_weight=base_codec svd_pre_weight=cls learning_rate=1e-2
pueue add -g left -l od-open -- python detection_lora_head.py port=5413 mode=TTT model=tic_svdlora quality_level=3 task_model=faster_rcnn transtic_weight=base_codec svd_pre_weight=cls learning_rate=1e-2
pueue add -g left -l od-open -- python detection_lora_head.py port=5414 mode=TTT model=tic_svdlora quality_level=4 task_model=faster_rcnn transtic_weight=base_codec svd_pre_weight=cls learning_rate=1e-2
