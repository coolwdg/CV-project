from PIL import Image
import torch
import open_clip
import numpy as np
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
from fer.fer import FER
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 表情描述 prompt 队列（按强度排序）
prompt_queues = {
    "happy": [
        "面带礼貌性微笑", "轻微的愉悦表情", "温柔的笑容", "若有所思的微笑",
        "明显的微笑", "愉快的表情", "轻松的笑容", "会心的微笑",
        "灿烂的笑容", "开怀大笑的脸", "洋溢着幸福的笑容", "笑得眼睛眯起来",
        "捧腹大笑", "喜极而泣的表情", "笑得前仰后合", "狂喜的表情"
    ],
    "angry": [
        "轻微的不悦", "皱起眉头的表情", "有些生气的脸", "紧闭嘴巴显得压抑的愤怒",
        "明显的怒气", "瞪眼怒视", "咬牙切齿的愤怒", "愤怒到脸色发红", "怒吼的脸"
    ],
    "sad": [
        "若有所思的忧郁", "微微皱眉的悲伤", "略显失落的脸", "眼神空洞的悲伤",
        "泪眼朦胧", "明显的沮丧", "痛苦的表情", "流泪的脸", "哭泣不止"
    ],
    "surprise": [
        "微微张嘴", "轻度惊讶", "略显意外的神情", "突然睁大眼睛",
        "明显的惊讶", "吃惊的表情", "惊讶得后仰", "瞠目结舌", "大吃一惊"
    ],
    "fear": [
        "警惕的表情", "神情紧张", "略显惊恐", "眼神游离",
        "张口结舌的恐惧", "大幅皱眉", "面容扭曲的恐惧", "歇斯底里的脸", "吓得不知所措"
    ],
    "disgust": [
        "轻微的厌烦", "皱鼻子的表情", "有些反感的脸", "撇嘴的神情",
        "明显的厌恶", "不屑的眼神", "做出嫌弃表情", "极度反感", "恶心想吐"
    ]
}

# 支持的主情绪标签
valid_emotions = set(prompt_queues.keys())

# 图片路径列表
image_paths = []

# 初始化模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

clip_model, _, processor = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
clip_model = clip_model.to(device).eval()

text_tokenizer = BertTokenizer.from_pretrained("./IDEA-CCNL_Taiyi-CLIP-RoBERTa-326M-ViT-H-Chinese")
text_encoder = BertModel.from_pretrained("./IDEA-CCNL_Taiyi-CLIP-RoBERTa-326M-ViT-H-Chinese").to(device).eval()

emotion_detector = FER(mtcnn=True)

# 主流程
results = []

for path in image_paths:
    try:
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)

        # Step 1: 表情识别
        detection = emotion_detector.detect_emotions(img_np)
        if detection:
            dominant_emotion = max(detection[0]['emotions'].items(), key=lambda x: x[1])[0]
        else:
            dominant_emotion = "happy"  # fallback 默认快乐

        print(f"{os.path.basename(path)} 粗识别情绪: {dominant_emotion}")

        if dominant_emotion not in valid_emotions:
            dominant_emotion = "happy"

        prompts = prompt_queues[dominant_emotion]

        # Step 2: 文本编码
        with torch.no_grad():
            text_inputs = text_tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)
            text_features = text_encoder(**text_inputs)[1]
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Step 3: 图像编码
        with torch.no_grad():
            image_tensor = processor(img).unsqueeze(0).to(device)
            image_features = clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            logit_scale = clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            probs = logits.softmax(dim=-1).cpu().numpy().squeeze()

        topk = np.argsort(probs)[-3:][::-1]
        top_prompts = [prompts[i] for i in topk]
        top_scores = [probs[i] for i in topk]

        results.append((img, dominant_emotion, top_prompts, top_scores))

    except Exception as e:
        print(f"跳过图片 {path}, 错误: {e}")

# 可视化
fig, axes = plt.subplots(len(results), 2, figsize=(14, 5 * len(results)))
if len(results) == 1:
    axes = [axes]

for i, (img, emo, top_prompts, top_scores) in enumerate(results):
    axes[i][0].imshow(img)
    axes[i][0].axis("off")
    axes[i][0].set_title(f"粗情绪识别: {emo}", fontsize=16)

    text = "Top-3 精细匹配:\n"
    for j, (p, s) in enumerate(zip(top_prompts, top_scores), 1):
        text += f"{j}. {p}   ({s:.3f})\n"
    axes[i][1].axis("off")
    axes[i][1].text(0.01, 0.95, text.strip(), fontsize=14, verticalalignment='top')

plt.tight_layout()
plt.show()
