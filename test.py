# # # from tokenizers.implementations import ByteLevelBPETokenizer
# # # from torch.utils.data import DataLoader

# # # from dataset import CNNDailyMailDataset, DynamicBatchSampler, collate_fn
# # # from neural_intra_attention_model import NeuralIntraAttentionModel
# # # from pointer_generator_network import PointerGeneratorNetwork
# # # from tokenization import PointerGeneratorTokenizer
# # # from transformer import Transformer
# # # from utils import set_seed

# # # set_seed()

# # # # tokenizer = PointerGeneratorTokenizer("word_level_vocab.json")
# # # tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")

# # # ds = CNNDailyMailDataset(
# # #     split="train",
# # #     tokenizer=tokenizer,
# # #     dataset_length=None,
# # # )
# # # loader = DataLoader(
# # #     ds,
# # #     collate_fn=collate_fn,
# # #     batch_sampler=DynamicBatchSampler(ds, max_tokens=10000, shuffle=True),
# # # )

# # # model = Transformer(
# # #     tokenizer=tokenizer,
# # #     d_model=128,
# # #     nhead=8,
# # #     num_layers=2,
# # #     learning_rate=1e-3,
# # #     device="cpu",
# # # )

# # # for batch_idx, batch in enumerate(loader):
# # #     batch_output_ids = model.infer(
# # #         batch["input_ids"],
# # #         max_output_length=10,
# # #         beam_width=4,
# # #         return_embedding=True,
# # #         return_attention=True,
# # #     )["output_ids"]
# # #     print(batch_output_ids)

# # # import os
# # # import pickle
# # # import re
# # # import shutil
# # # from collections import defaultdict

# # # import matplotlib.pyplot as plt
# # # import torch
# # # from tokenizers.implementations import ByteLevelBPETokenizer
# # # from torch.utils.data import DataLoader

# # # from dataset import CNNDailyMailDataset, DynamicBatchSampler, build_collate_fn
# # # from environment import adaptive_display, detect_runtime_env, try_set_window_position
# # # from metrics import compute_metric
# # # from neural_intra_attention_model import NeuralIntraAttentionModel
# # # from pointer_generator_network import PointerGeneratorNetwork
# # # from tokenization import PointerGeneratorTokenizer
# # # from transformer import Transformer
# # # from utils import (
# # #     load_checkpoint,
# # #     name_to_latex,
# # #     print_log_file,
# # #     save_checkpoint,
# # #     set_seed,
# # #     token_ids_to_text,
# # # )

# # # set_seed()

# # # MODEL = "POINTER_GENERATOR_NETWORK"
# # # CHECKPOINT_FOLDER = f"{MODEL.lower()}_checkpoints"
# # # NUM_EPOCHS = 200
# # # MAX_TOKENS_EACH_BATCH = 60000
# # # TRAIN_DATASET_LENGTH = None
# # # VALIDATION_DATASET_LENGTH = None
# # # CONTINUE_TRAINING = True
# # # LAST_TRAIN_STEP_FILE = f"{CHECKPOINT_FOLDER}/last_train_step.pkl"
# # # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # # LOSS_LOG_MODE = "graph"
# # # LOSS_LOG_INTERVAL = 10
# # # ENV = detect_runtime_env()
# # # METRICS = ["rouge1", "rouge2", "rougeL", "bleu4", "meteor", "bertscore", "moverscore"]
# # # MODEL_SAVE_INTERVAL = 10
# # # CHECKPOINT_INTERVAL = 10

# # # if ENV in ("colab", "notebook"):
# # #     from IPython.display import clear_output, display


# # # def find_latest_checkpoint():
# # #     if os.path.exists(CHECKPOINT_FOLDER) and any(
# # #         re.match(r"^checkpoint([1-9]\d*)\.pt$", f)
# # #         for f in os.listdir(CHECKPOINT_FOLDER)
# # #     ):
# # #         latest_checkpoint_idx = max(
# # #             (
# # #                 int(m.group(1))
# # #                 for f in os.listdir(CHECKPOINT_FOLDER)
# # #                 if (m := re.match(r"^checkpoint([1-9]\d*)\.pt$", f))
# # #             )
# # #         )

# # #         return (
# # #             f"{CHECKPOINT_FOLDER}/checkpoint{latest_checkpoint_idx}.pt",
# # #             latest_checkpoint_idx,
# # #         )

# # #     return None, -1


# # # def clear_checkpoint_folder():
# # #     if os.path.exists(CHECKPOINT_FOLDER):
# # #         [
# # #             shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
# # #             for p in (
# # #                 os.path.join(CHECKPOINT_FOLDER, f)
# # #                 for f in os.listdir(CHECKPOINT_FOLDER)
# # #             )
# # #         ]
# # #     else:
# # #         os.makedirs(CHECKPOINT_FOLDER)


# # # if __name__ == "__main__":
# # #     if MODEL == "TRANSFORMER":
# # #         tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
# # #     else:
# # #         tokenizer = PointerGeneratorTokenizer("word_level_vocab.json")

# # #     ds = {
# # #         "train": CNNDailyMailDataset(
# # #             split="train",
# # #             tokenizer=tokenizer,
# # #             dataset_length=TRAIN_DATASET_LENGTH,
# # #         ),
# # #         "validation": CNNDailyMailDataset(
# # #             split="validation",
# # #             tokenizer=tokenizer,
# # #             dataset_length=VALIDATION_DATASET_LENGTH,
# # #         ),
# # #     }
# # #     loader = {
# # #         "train": DataLoader(
# # #             ds["train"],
# # #             collate_fn=build_collate_fn(tokenizer),
# # #             batch_sampler=DynamicBatchSampler(
# # #                 ds["train"],
# # #                 max_tokens=MAX_TOKENS_EACH_BATCH,
# # #             ),
# # #             pin_memory=True if DEVICE == "cuda" else False,
# # #         ),
# # #         "validation": DataLoader(
# # #             ds["validation"],
# # #             collate_fn=build_collate_fn(tokenizer),
# # #             batch_sampler=DynamicBatchSampler(
# # #                 ds["validation"],
# # #                 max_tokens=MAX_TOKENS_EACH_BATCH,
# # #             ),
# # #             pin_memory=True if DEVICE == "cuda" else False,
# # #         ),
# # #     }

# # #     for epoch in range(NUM_EPOCHS):
# # #         for batch_idx, batch in enumerate(loader["train"]):
# # #             print(
# # #                 f"{batch["input_ids"].shape[1]} | {batch["target_ids"].shape[1]} | {batch["input_ids"].shape[0]}"
# # #             )

# # import os
# # import pickle
# # import shutil
# # from collections import defaultdict

# # import matplotlib.pyplot as plt
# # import torch
# # from tokenizers.implementations import ByteLevelBPETokenizer

# # from dataset import (
# #     CNNDailyMailDataset,
# #     DataLoader,
# #     DynamicBatchSampler,
# #     build_collate_fn,
# # )
# # from environment import adaptive_display, detect_runtime_env, try_set_window_position
# # from neural_intra_attention_model import NeuralIntraAttentionModel
# # from pointer_generator_network import PointerGeneratorNetwork
# # from tokenization import PointerGeneratorTokenizer
# # from transformer import Transformer
# # from utils import load_checkpoint, name_to_latex, save_checkpoint, set_seed

# # set_seed()

# # MODEL = "TRANSFORMER"
# # CHECKPOINT_FOLDER = f"{MODEL.lower()}_checkpoints"
# # NUM_EPOCHS = 200
# # MAX_TOKENS_EACH_BATCH = 10000
# # TRAIN_DATASET_LENGTH = 100
# # VALIDATION_DATASET_LENGTH = 100
# # CONTINUE_TRAINING = True
# # TEMP_MODEL_FILE = f"{CHECKPOINT_FOLDER}/temp_model.pt"
# # LAST_TRAIN_STEP_FILE = f"{CHECKPOINT_FOLDER}/last_train_step.pkl"
# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # LOSS_LOG_MODE = "graph"
# # LOSS_LOG_INTERVAL = 10
# # ENV = detect_runtime_env()
# # MODEL_SAVE_INTERVAL = 10

# # if ENV in ("colab", "notebook"):
# #     from IPython.display import clear_output, display


# # def clear_checkpoint_folder():
# #     if os.path.exists(CHECKPOINT_FOLDER):
# #         [
# #             shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
# #             for p in (
# #                 os.path.join(CHECKPOINT_FOLDER, f)
# #                 for f in os.listdir(CHECKPOINT_FOLDER)
# #             )
# #         ]
# #     else:
# #         os.makedirs(CHECKPOINT_FOLDER)


# # if __name__ == "__main__":
# #     loss_log_file = open("loss_log.txt", "w")

# #     if MODEL == "TRANSFORMER":
# #         tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")
# #     else:
# #         tokenizer = PointerGeneratorTokenizer("word_level_vocab.json")

# #     collate_fn = build_collate_fn(tokenizer)
# #     ds = {
# #         "train": CNNDailyMailDataset(
# #             split="train",
# #             tokenizer=tokenizer,
# #             dataset_length=TRAIN_DATASET_LENGTH,
# #         ),
# #         "validation": CNNDailyMailDataset(
# #             split="validation",
# #             tokenizer=tokenizer,
# #             dataset_length=VALIDATION_DATASET_LENGTH,
# #         ),
# #     }
# #     loader = {
# #         "train": DataLoader(
# #             ds["train"],
# #             collate_fn=collate_fn,
# #             batch_sampler=DynamicBatchSampler(
# #                 ds["train"],
# #                 max_tokens=MAX_TOKENS_EACH_BATCH,
# #             ),
# #         ),
# #         "validation": DataLoader(
# #             ds["validation"],
# #             collate_fn=collate_fn,
# #             batch_sampler=DynamicBatchSampler(
# #                 ds["validation"],
# #                 max_tokens=MAX_TOKENS_EACH_BATCH,
# #             ),
# #         ),
# #     }

# #     match MODEL:
# #         case "POINTER_GENERATOR_NETWORK":
# #             model = PointerGeneratorNetwork(
# #                 tokenizer=tokenizer,
# #                 embedding_dim=128,
# #                 encoder_hidden_dim=256,
# #                 decoder_hidden_dim=256,
# #                 attention_dim=256,
# #                 bottle_neck_dim=512,
# #                 num_layers=2,
# #                 cov_loss_factor=1.0,
# #                 learning_rate=1e-3,
# #                 device=DEVICE,
# #             )
# #         case "NEURAL_INTRA_ATTENTION_MODEL":
# #             model = NeuralIntraAttentionModel(
# #                 tokenizer=tokenizer,
# #                 embedding_dim=128,
# #                 hidden_dim=256,
# #                 bottle_neck_dim=512,
# #                 num_layers=2,
# #                 rl_loss_factor=0.0,
# #                 learning_rate=1e-3,
# #                 device=DEVICE,
# #             )
# #         case "TRANSFORMER":
# #             model = Transformer(
# #                 tokenizer=tokenizer,
# #                 d_model=128,
# #                 nhead=8,
# #                 num_layers=2,
# #                 learning_rate=1e-3,
# #                 device=DEVICE,
# #             )

# #     def return_default_dict_list():
# #         return defaultdict(list)

# #     epoch_loss_history = defaultdict(return_default_dict_list)
# #     batch_loss_history = defaultdict(list)

# #     if CONTINUE_TRAINING and os.path.exists(TEMP_MODEL_FILE):
# #         load_checkpoint(model, TEMP_MODEL_FILE, map_location=DEVICE)
# #         print("Model loaded successfully!")
# #         with open(LAST_TRAIN_STEP_FILE, "rb") as f:
# #             (
# #                 latest_epoch_idx,
# #                 latest_batch_idx,
# #                 epoch_loss_history,
# #                 batch_loss_history,
# #                 latest_epoch_num_tokens,
# #                 latest_raw_batch_loss_history,
# #                 latest_num_samples,
# #             ) = pickle.load(f)
# #     else:
# #         clear_checkpoint_folder()
# #         CONTINUE_TRAINING = False

# #     for epoch in range(NUM_EPOCHS):
# #         if CONTINUE_TRAINING and epoch < latest_epoch_idx:
# #             continue
# #         for split in ["train", "validation"]:
# #             batch_step = (
# #                 model.train_one_batch if split == "train" else model.validate_one_batch
# #             )
# #             if CONTINUE_TRAINING:
# #                 epoch_num_tokens = latest_epoch_num_tokens
# #                 num_samples = latest_num_samples
# #                 raw_batch_loss_history = latest_raw_batch_loss_history
# #                 loader[split].skip_batches = latest_batch_idx + 1
# #                 CONTINUE_TRAINING = False
# #             else:
# #                 epoch_num_tokens = 0
# #                 raw_batch_loss_history = defaultdict(list)
# #                 num_samples = 0
# #                 loader[split].skip_batches = 0
# #                 latest_batch_idx = -1
# #             for batch_idx, batch in enumerate(loader[split]):
# #                 batch_idx = batch_idx + latest_batch_idx + 1
# #                 batch_num_tokens = batch["target_length"].sum().item()
# #                 epoch_num_tokens += batch_num_tokens
# #                 num_samples += len(batch["input_ids"])
# #                 match MODEL:
# #                     case "POINTER_GENERATOR_NETWORK":
# #                         losses = batch_step(
# #                             batch["input_ids"],
# #                             batch["target_ids"],
# #                             batch["oov_list"],
# #                             batch["input_length"],
# #                             batch["target_length"],
# #                         )
# #                     case "NEURAL_INTRA_ATTENTION_MODEL":
# #                         model.rl_loss_factor = 3.0 if epoch > 2 else 0.0
# #                         losses = batch_step(
# #                             batch["input_ids"],
# #                             batch["target_ids"],
# #                             batch["oov_list"],
# #                             batch["input_length"],
# #                             batch["target_length"],
# #                             target_texts=batch["target_text"],
# #                         )
# #                     case "TRANSFORMER":
# #                         losses = batch_step(
# #                             batch["input_ids"],
# #                             batch["target_ids"],
# #                         )

# #                 for loss_type, loss_value in losses.items():
# #                     if split == "train":
# #                         batch_loss_history[loss_type].append(
# #                             loss_value / batch_num_tokens
# #                         )
# #                     raw_batch_loss_history[loss_type].append(loss_value)

# #                 if split == "train" and batch_idx % LOSS_LOG_INTERVAL == 0:
# #                     print(f"{num_samples}/{len(ds['train'])} samples")
# #                     average_loss_per_token = losses["total_loss"] / batch_num_tokens
# #                     log = f"Epoch {epoch} / Batch {batch_idx} : Average Loss Per Token is {average_loss_per_token}"
# #                     match LOSS_LOG_MODE:
# #                         case "console":
# #                             print(log)
# #                         case "file":
# #                             loss_log_file.write(log + "\n")
# #                         case "graph":
# #                             for (
# #                                 loss_type,
# #                                 loss_values,
# #                             ) in batch_loss_history.items():
# #                                 if batch_line_2ds[loss_type] is None:
# #                                     batch_line_2ds[loss_type] = batch_ax.plot(
# #                                         loss_values,
# #                                         label=name_to_latex[loss_type],
# #                                     )[0]
# #                                     batch_ax.legend()
# #                                 else:
# #                                     batch_line_2ds[loss_type].set_xdata(
# #                                         range(len(loss_values))
# #                                     )
# #                                     batch_line_2ds[loss_type].set_ydata(loss_values)

# #                             batch_ax.relim()
# #                             batch_ax.autoscale()
# #                             figure.canvas.draw()
# #                             figure.canvas.flush_events()
# #                             if ENV in ("colab", "notebook"):
# #                                 clear_output(wait=True)
# #                                 display(figure)

# #                 if split == "train" and batch_idx % MODEL_SAVE_INTERVAL == 0:
# #                     save_checkpoint(model, TEMP_MODEL_FILE)

# #                     with open(LAST_TRAIN_STEP_FILE, "wb") as f:
# #                         pickle.dump(
# #                             (
# #                                 epoch,
# #                                 batch_idx,
# #                                 epoch_loss_history,
# #                                 batch_loss_history,
# #                                 epoch_num_tokens,
# #                                 raw_batch_loss_history,
# #                                 num_samples,
# #                             ),
# #                             f,
# #                         )

# #                     if ENV == "colab" and os.path.exists("/content/drive/MyDrive"):
# #                         save_checkpoint(model, f"/content/drive/MyDrive/temp_model.pt")
# #                         with open(
# #                             f"/content/drive/MyDrive/last_train_step.pkl", "wb"
# #                         ) as f:
# #                             pickle.dump(
# #                                 (
# #                                     epoch,
# #                                     batch_idx,
# #                                     epoch_loss_history,
# #                                     batch_loss_history,
# #                                     epoch_num_tokens,
# #                                     raw_batch_loss_history,
# #                                     num_samples,
# #                                 ),
# #                                 f,
# #                             )

# #                     print("Model saved successfully!")

# #                 if split == "validation":
# #                     print(
# #                         f"Validated {num_samples}/{VALIDATION_DATASET_LENGTH if VALIDATION_DATASET_LENGTH is not None else len(ds['validation'])} samples"
# #                     )

# #                 torch.cuda.empty_cache()

# #             for loss_type, loss_values in raw_batch_loss_history.items():
# #                 epoch_loss_history[split][loss_type].append(
# #                     sum(loss_values) / epoch_num_tokens
# #                 )

# #             log = f"Epoch {epoch} / {split.capitalize()} : Average Loss Per Token is {epoch_loss_history[split]['total_loss'][-1]}"

# #             match LOSS_LOG_MODE:
# #                 case "console":
# #                     print(log)
# #                 case "file":
# #                     loss_log_file.write(log + "\n")
# #                 case "graph":
# #                     for (
# #                         loss_type,
# #                         loss_values,
# #                     ) in epoch_loss_history[split].items():
# #                         if epoch_line_2ds[split][loss_type] is None:
# #                             epoch_line_2ds[split][loss_type] = epoch_ax.plot(
# #                                 loss_values,
# #                                 label=name_to_latex[loss_type] + f" ({split})",
# #                             )[0]
# #                             epoch_ax.legend()
# #                         else:
# #                             epoch_line_2ds[split][loss_type].set_xdata(
# #                                 range(len(loss_values))
# #                             )
# #                             epoch_line_2ds[split][loss_type].set_ydata(loss_values)

# #                     epoch_ax.relim()
# #                     epoch_ax.autoscale()
# #                     figure.canvas.draw()
# #                     figure.canvas.flush_events()
# #                     if ENV in ("colab", "notebook"):
# #                         clear_output(wait=True)
# #                         display(figure)

# #             if split == "train":
# #                 save_checkpoint(model, f"{CHECKPOINT_FOLDER}/checkpoint_{epoch}.pt")
# #                 if ENV == "colab" and os.path.exists("/content/drive/MyDrive"):
# #                     save_checkpoint(
# #                         model, f"/content/drive/MyDrive/checkpoint_{epoch}.pt"
# #                     )

# #     loss_log_file.close()
# #     if LOSS_LOG_MODE == "graph" and ENV == "gui":
# #         plt.show()

# from joblib import Memory

# memory = Memory("cache_dir", verbose=0)

# print(memory.cache(lambda: (print("Computing..."), 9)[1])())
# print(memory.cache(lambda: (print("Computing..."), 9)[1])())
import pickle

with open("cache/train_batches.pkl", "rb") as f:
    a, b = pickle.load(f)
    a = a[:32361]
    #     print(a[32360][-1])
    with open("cache/train_batches2.pkl", "wb") as f2:
        pickle.dump((a, 264229), f2)
#     flat_A = [y for x in a for y in x]
#     # flat_A = flat_A[263993]
#     print(len(flat_A))
#     print(len(set(flat_A)))
#     print(flat_A.index(256466))
#     71349
#     print([i for i, x in enumerate(a) for y in x if y == 256466])
#     32361
#     256466
#     32341
#     263759
#     19868 tìm index của nó trong indices + 1 => gán b
# 264229

# with open("cache/train_sorted_indices.pkl", "rb") as f:
#     a = pickle.load(f)
#     print(a[0].index(19868))
