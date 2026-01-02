from transformers import AutoTokenizer, AutoModelForCausalLM, MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
import re
import torch
import argparse
from read_data import get_dataset
from PIL import Image
from tqdm import tqdm, trange
import json
from sklearn.metrics import f1_score, accuracy_score

import logging
logging.disable(logging.WARNING)


def encode_one_sample(sample):
    claim_id = sample[0]
    claim = sample[1]
    text_evidence = sample[2]
    image_evidence = sample[3]
    label = sample[4]
    

    encoded_sample = {}
    encoded_sample["claim_id"] = str(claim_id)
    encoded_sample["claim"] = claim
    encoded_sample["label"] = label
    encoded_sample['text_evidence'] = text_evidence
    encoded_sample['image_evidence'] = image_evidence

    return encoded_sample


class ClaimVerificationDataset(torch.utils.data.Dataset):
    def __init__(self, claim_verification_data):
        self._data = claim_verification_data
        # self._processor = processor

        self._encoded = []
        for d in self._data:
            self._encoded.append(encode_one_sample(d))

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]

    def to_list(self):
        return self._encoded


def load_peft_model_vision(peft_model_name, device="auto"):
    processor = AutoProcessor.from_pretrained(
        peft_model_name,
        model_max_length=2000,
        padding_side="left",
        truncation_side="left",
        token="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",
    )

    quantization_config = BitsAndBytesConfig(
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        load_in_4bit=True,
    )

    model = MllamaForConditionalGeneration.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="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",
        device_map="auto",
        use_flash_attention_2=False,
    )

    return processor, model


def make_prompt(text_evidence):
    prompt = f"""
    <|image|><|begin_of_text|>{text_evidence} \n

    Please generate a short paragraph describing the about the consistency of the image based on the given text following this template:
    \n
    <HYPOTHESIS>: Please determining whether the image is consistent with the text or not.
    <EXPLANATION>: Explanation the aligment between the image hypothesis and the text.
    <FINAL ANSWER>: Give one paragraph describing the consistency of the image and text based on the explanation.
    """

    return prompt


@torch.inference_mode()
def do_inference_vision(model, processor, prompt, image):
    image_data = Image.open(image)

    inputs = processor(image_data, prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,    
        top_p=0.5,
    )
    return processor.decode(output_ids[0])


def create_align_form(dataset, model, processor, path):
    def get_image_path_new(img_p, path):
        name = img_p.split('/')[-1]
        return path + "/images/" + name
    
    results = []
    print("---performing.....----")
    for sample in tqdm(dataset):
        align_text_image = []
        if len(sample['image_evidence']) > 0:
            for ie in sample['image_evidence']:
                for te in sample['text_evidence']:
                    img = get_image_path_new(ie, path)
                    try:
                        align_text_image.append({
                            'text': te,
                            'image': ie,
                            'alignment': do_inference_vision(model, processor, make_prompt(te), img)
                        })
                    except Exception as e:
                        print(e)
        results.append({
            **sample,
            'alignment': align_text_image
        })

    return results


def read_augmented_data(data):
    for d in data:
        if len(d['alignment']) > 0:
            for a in d['alignment']:
                temp = a['alignment'].split("\n\n\n\n")[-1]
                a['clean_alignment'] = temp.replace('<|eot_id|>', '').strip()
    return data



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/s2320014/data")
    parser.add_argument('--limit', default=False, action='store_true')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    processor, model = load_peft_model_vision("meta-llama/Llama-3.2-90B-Vision-Instruct")
    
    print("Running on gold evidence")
    dataset = get_dataset(args.path)
    data_claim = ClaimVerificationDataset(dataset)

    print("loadded: {}".format(len(data_claim)))

    if args.limit:
        print("Run with limit: {}-{}".format(args.start, args.end))
        data_claim = data_claim[args.start:args.end]

    result_dev = create_align_form(data_claim, model, processor, args.path)
    with open('./finfact_claim_llama3.2_full.json' if not args.limit else './finfact_claim_llama3.2_full_{}-{}.json'.format(args.start, args.end), 'w', encoding='utf-8') as f:
        json.dump(result_dev, f, ensure_ascii=False, indent=4)
    f.close()
