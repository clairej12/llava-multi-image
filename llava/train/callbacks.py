from transformers import TrainerCallback
import os
import json
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images

from PIL import Image
import math

from llava.train.utils import tensor_to_pil


class ImageEvalCallback(TrainerCallback):
    def __init__(self, save_dir, question_file, eval_steps=5):
        """
        eval_steps: int
            Evaluate every `eval_steps` steps.
        """
        self.eval_steps = eval_steps
        self.save_dir = save_dir
        self.question_file = question_file

    def on_step_end(self, args, state, control, **kwargs):
        """
        This method will be called at the end of each training step.
        """

        if state.global_step % self.eval_steps == 0 and state.is_local_process_zero:
            model = kwargs.get('model', None)
            tokenizer = kwargs.get('tokenizer', None)
            image_processor = model.get_vision_tower().image_processor

            root = os.path.join(self.save_dir, "multi_image_log", f'{state.global_step:06}')
            os.makedirs(root, exist_ok=True)
            
            questions = [json.loads(q) for q in open(os.path.expanduser(self.question_file), "r")]
            answers = []
            for line in questions:
                idx = line["question_id"]
                image_file = line["image"]
                qs = line["text"]
                cur_prompt = qs
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
                image_tensor = process_images([image], image_processor, model.config)[0]

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        use_cache=True)

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                
                answers.append(json.dumps({"question_id": idx,
                                        "prompt": cur_prompt,
                                        "text": outputs,
                                        "metadata": {}}) + "\n")

            # Save Prompts
            filename = "answers.json"
            path = os.path.join(root, filename)
            with open(path, "w") as f:
                for p in answers:
                    f.write(f"{json.dumps(p)}\n\n")